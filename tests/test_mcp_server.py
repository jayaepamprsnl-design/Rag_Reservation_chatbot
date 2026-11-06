"""Tests for app/mcp_server/server.py (store logic + HTTP/auth layer)."""
import json
import logging
import re
import threading

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from app.mcp_server.server import (
    HEADER,
    McpSettings,
    ReservationRecord,
    ReservationStore,
    build_app,
)

TOKEN = "test-token-" + "x" * 24
LINE_RE = re.compile(r"^[^|\r\n]+ \| [A-Z0-9\-]+ \| \d{4}-\d\d-\d\d \d\d:\d\d - .+ \| \d{4}-\d\d-\d\d \d\d:\d\d:\d\d$")


def rec(**over):
    base = dict(
        reservation_id=1,
        customer_name="jaya kumari",
        car_plate="TS09AB1234",
        start_time="2026-11-04T14:00:00",
        end_time="2026-11-04T17:00:00",
        approved_at="2026-11-03T11:30:00",
    )
    base.update(over)
    return base


@pytest.fixture
def store(tmp_path):
    return ReservationStore(tmp_path / "out" / "nested" / "confirmed.txt")


def lines(store):
    return store.path.read_text(encoding="utf-8").split("\n")


# ── line format ──────────────────────────────────────────────────────────────


class TestLineFormat:
    def test_exact_line_matches_spec_example(self, store):
        store.write(ReservationRecord(**rec()))
        assert lines(store)[1] == (
            "jaya kumari | TS09AB1234 | 2026-11-04 14:00 - 17:00 | 2026-11-03 11:30:00"
        )

    def test_multi_day_period_shows_both_dates(self):
        r = ReservationRecord(**rec(end_time="2026-11-05T10:00:00"))
        assert "2026-11-04 14:00 - 2026-11-05 10:00" in r.to_line()

    def test_plate_is_uppercased_and_name_trimmed(self):
        r = ReservationRecord(**rec(car_plate=" ts09ab1234 ", customer_name="  Ann  "))
        assert r.to_line().startswith("Ann | TS09AB1234 | ")

    def test_unicode_name_is_utf8_roundtrip(self, store):
        store.write(ReservationRecord(**rec(customer_name="Zoë Müller 山田")))
        assert "Zoë Müller 山田 | " in store.path.read_text(encoding="utf-8")

    def test_lf_only_line_endings(self, store):
        store.write(ReservationRecord(**rec()))
        assert b"\r" not in store.path.read_bytes()


# ── file creation / header ───────────────────────────────────────────────────


class TestFileCreation:
    def test_directory_and_file_created_on_first_write(self, store):
        assert not store.path.parent.exists()
        store.write(ReservationRecord(**rec()))
        assert store.path.is_file()

    def test_header_is_first_line(self, store):
        store.write(ReservationRecord(**rec()))
        assert lines(store)[0] == HEADER

    def test_header_written_once_only(self, store):
        for i in (1, 2, 3):
            store.write(ReservationRecord(**rec(reservation_id=i)))
        assert lines(store).count(HEADER) == 1
        assert len([ln for ln in lines(store) if ln]) == 4

    def test_no_file_before_first_write_and_empty_tail(self, store):
        assert not store.path.exists()
        assert store.tail(5) == []


# ── idempotency ──────────────────────────────────────────────────────────────


class TestDuplicate:
    def test_second_write_same_id_is_duplicate(self, store):
        assert store.write(ReservationRecord(**rec())) == "written"
        assert store.write(ReservationRecord(**rec())) == "duplicate"

    def test_duplicate_adds_no_line_even_if_fields_differ(self, store):
        store.write(ReservationRecord(**rec()))
        assert store.write(ReservationRecord(**rec(customer_name="Someone Else"))) == "conflict"
        assert len([ln for ln in lines(store) if ln]) == 2  # header + 1
        assert "Someone Else" not in store.path.read_text(encoding="utf-8")

    def test_duplicate_detected_after_restart(self, store):
        store.write(ReservationRecord(**rec()))
        fresh = ReservationStore(store.path)
        assert fresh.write(ReservationRecord(**rec())) == "duplicate"

    def test_different_ids_both_written(self, store):
        assert store.write(ReservationRecord(**rec(reservation_id=1))) == "written"
        assert store.write(ReservationRecord(**rec(reservation_id=2))) == "written"


# ── injection ────────────────────────────────────────────────────────────────


class TestInjection:
    @pytest.mark.parametrize(
        "name",
        [
            "Eve | FAKE-1 | 2026-01-01 10:00 - 11:00 | 2026-01-01 09:00:00",  # forge columns
            "Eve\nMallory | X | y | z",                                      # new line
            "Eve\r\nMallory",                                                # CRLF
            "Eve\rMallory",
            "Eve\u2028Mallory",                                              # unicode line sep
            "Eve\u2029Mallory",
            "Eve\x00",                                                       # NUL
            "Eve\tTab",
            "Eve\x1b[31m",                                                   # ANSI escape
            "Eve\ufeff",                                                     # BOM / format char
        ],
    )
    def test_name_with_separator_or_control_chars_rejected(self, name):
        with pytest.raises(ValidationError):
            ReservationRecord(**rec(customer_name=name))

    @pytest.mark.parametrize("plate", ["AB|1234", "AB1234\nX", "AB 1234", "AB1234;DROP"])
    def test_plate_with_unsafe_chars_rejected(self, plate):
        with pytest.raises(ValidationError):
            ReservationRecord(**rec(car_plate=plate))

    def test_rejected_input_never_touches_file(self, store):
        with pytest.raises(ValidationError):
            ReservationRecord(**rec(customer_name="a|b"))
        assert not store.path.exists()

    def test_every_written_line_has_exactly_three_separators(self, store):
        store.write(ReservationRecord(**rec()))
        assert lines(store)[1].count(" | ") == 3


# ── input validation ─────────────────────────────────────────────────────────


class TestInvalidInput:
    @pytest.mark.parametrize("rid", [0, -1, 2**31])
    def test_bad_reservation_id(self, rid):
        with pytest.raises(ValidationError):
            ReservationRecord(**rec(reservation_id=rid))

    @pytest.mark.parametrize("name", ["", "   ", "x" * 101])
    def test_bad_name_length(self, name):
        with pytest.raises(ValidationError):
            ReservationRecord(**rec(customer_name=name))

    @pytest.mark.parametrize("plate", ["", "A", "AB", "-ABC", "ABC-", "A" * 21])
    def test_bad_plate(self, plate):
        with pytest.raises(ValidationError):
            ReservationRecord(**rec(car_plate=plate))

    @pytest.mark.parametrize("field", ["start_time", "end_time", "approved_at"])
    @pytest.mark.parametrize("bad", ["tomorrow", "2026-13-45T00:00:00", "", "9" * 60])
    def test_bad_datetime(self, field, bad):
        with pytest.raises(ValidationError):
            ReservationRecord(**rec(**{field: bad}))

    def test_end_must_be_after_start(self):
        with pytest.raises(ValidationError):
            ReservationRecord(**rec(end_time="2026-11-04T14:00:00"))
        with pytest.raises(ValidationError):
            ReservationRecord(**rec(end_time="2026-11-04T13:00:00"))

    def test_mixed_naive_and_aware_rejected(self):
        with pytest.raises(ValidationError):
            ReservationRecord(**rec(end_time="2026-11-04T17:00:00+00:00"))


# ── concurrency ──────────────────────────────────────────────────────────────


class TestConcurrency:
    def _run(self, targets):
        errors = []

        def wrap(fn):
            def go():
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
            return go

        threads = [threading.Thread(target=wrap(t)) for t in targets]
        barrier_start = threading.Event()
        for t in threads:
            t.start()
        barrier_start.set()
        for t in threads:
            t.join(timeout=60)
        assert not errors, errors

    def test_20_threads_distinct_ids_no_lost_or_interleaved_lines(self, store):
        def job(i):
            return lambda: store.write(
                ReservationRecord(**rec(reservation_id=i, customer_name=f"Customer {i:02d}"))
            )

        self._run([job(i) for i in range(1, 21)])
        body = [ln for ln in lines(store) if ln]
        assert body[0] == HEADER
        data = body[1:]
        assert len(data) == 20
        assert all(LINE_RE.match(ln) for ln in data), data
        assert {ln.split(" | ")[0] for ln in data} == {f"Customer {i:02d}" for i in range(1, 21)}
        assert lines(store).count(HEADER) == 1

    def test_20_threads_same_id_written_exactly_once(self, store):
        results = []
        self._run([lambda: results.append(store.write(ReservationRecord(**rec()))) for _ in range(20)])
        assert results.count("written") == 1
        assert results.count("duplicate") == 19
        assert len([ln for ln in lines(store) if ln]) == 2

    def test_two_store_instances_share_os_lock(self, store):
        other = ReservationStore(store.path)  # simulates a second process
        targets = []
        for i in range(1, 21):
            s = store if i % 2 else other
            targets.append(
                lambda s=s, i=i: s.write(ReservationRecord(**rec(reservation_id=i)))
            )
        self._run(targets)
        data = [ln for ln in lines(store) if ln][1:]
        assert len(data) == 20 and all(LINE_RE.match(ln) for ln in data)


# ── HTTP layer: auth, rate limit, health, tools ──────────────────────────────

HDRS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def settings(tmp_path, **over):
    return McpSettings(
        _env_file=None,
        mcp_api_key=over.pop("mcp_api_key", TOKEN),
        reservations_file=str(tmp_path / "http" / "confirmed.txt"),
        **over,
    )


def rpc(client, method, params, token=TOKEN, scheme="Bearer"):
    headers = dict(HDRS)
    if token is not None:
        headers["Authorization"] = f"{scheme} {token}"
    return client.post(
        "/mcp", headers=headers,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
    )


def call_tool(client, name, args, **kw):
    return rpc(client, "tools/call", {"name": name, "arguments": args}, **kw)


@pytest.fixture
def client(tmp_path):
    app = build_app(settings(tmp_path))
    with TestClient(app) as c:
        yield c


class TestAuth:
    def test_missing_token_is_401(self, client):
        r = call_tool(client, "list_confirmed_reservations", {}, token=None)
        assert r.status_code == 401
        assert r.headers["www-authenticate"] == "Bearer"

    def test_wrong_token_is_401(self, client):
        assert call_tool(client, "list_confirmed_reservations", {}, token="nope").status_code == 401

    def test_token_of_same_length_but_wrong_is_401(self, client):
        assert call_tool(client, "list_confirmed_reservations", {}, token="y" * len(TOKEN)).status_code == 401

    def test_wrong_scheme_is_401(self, client):
        assert call_tool(client, "list_confirmed_reservations", {}, scheme="Basic").status_code == 401

    def test_valid_token_is_accepted(self, client):
        assert call_tool(client, "list_confirmed_reservations", {"limit": 5}).status_code == 200

    def test_unauthorized_write_creates_no_file(self, tmp_path, client):
        call_tool(client, "write_reservation", rec(), token="bad")
        assert not (tmp_path / "http").exists()

    def test_health_needs_no_token(self, client):
        r = client.get("/health")
        assert r.status_code == 200 and r.json() == {"status": "ok"}

    def test_server_refuses_to_start_without_api_key(self, tmp_path):
        with pytest.raises(RuntimeError, match="MCP_API_KEY"):
            build_app(settings(tmp_path, mcp_api_key=""))

    def test_default_bind_is_loopback(self):
        assert McpSettings(_env_file=None).mcp_host == "127.0.0.1"
        assert McpSettings(_env_file=None).mcp_port == 8765


class TestRateLimit:
    def test_exceeding_limit_returns_429(self, tmp_path):
        app = build_app(settings(tmp_path, mcp_rate_limit_per_minute=3))
        with TestClient(app) as c:
            codes = [call_tool(c, "list_confirmed_reservations", {}).status_code for _ in range(5)]
        assert codes[:3] == [200, 200, 200]
        assert codes[3:] == [429, 429]

    def test_rate_limit_also_applies_to_bad_tokens(self, tmp_path):
        app = build_app(settings(tmp_path, mcp_rate_limit_per_minute=2))
        with TestClient(app) as c:
            codes = [call_tool(c, "x", {}, token="bad").status_code for _ in range(4)]
        assert codes == [401, 401, 429, 429]

    def test_health_is_not_rate_limited(self, tmp_path):
        app = build_app(settings(tmp_path, mcp_rate_limit_per_minute=1))
        with TestClient(app) as c:
            assert all(c.get("/health").status_code == 200 for _ in range(5))


def tool_result(resp):
    body = resp.json()
    return body["result"]


class TestTools:
    def test_write_via_mcp_and_file_content(self, tmp_path, client):
        r = call_tool(client, "write_reservation", rec())
        res = tool_result(r)
        assert not res.get("isError")
        assert json.loads(res["content"][0]["text"]) == {"status": "written", "reservation_id": 1}
        text = (tmp_path / "http" / "confirmed.txt").read_text(encoding="utf-8")
        assert "jaya kumari | TS09AB1234 | 2026-11-04 14:00 - 17:00 | 2026-11-03 11:30:00" in text

    def test_duplicate_via_mcp(self, client):
        call_tool(client, "write_reservation", rec())
        res = tool_result(call_tool(client, "write_reservation", rec()))
        assert json.loads(res["content"][0]["text"]) == {"status": "duplicate", "reservation_id": 1}

    def test_injection_via_mcp_is_error_and_writes_nothing(self, tmp_path, client):
        res = tool_result(call_tool(client, "write_reservation", rec(customer_name="a | b\nc")))
        assert res["isError"] is True
        assert "must not contain '|'" in res["content"][0]["text"]
        assert not (tmp_path / "http").exists()

    def test_invalid_input_error_does_not_echo_value(self, client):
        res = tool_result(call_tool(client, "write_reservation", rec(car_plate="SECRET|PLATE")))
        assert res["isError"] is True
        assert "SECRET" not in res["content"][0]["text"]

    def test_list_returns_written_lines(self, client):
        call_tool(client, "write_reservation", rec(reservation_id=1))
        call_tool(client, "write_reservation", rec(reservation_id=2, customer_name="Bob"))
        res = tool_result(call_tool(client, "list_confirmed_reservations", {"limit": 1}))
        payload = json.loads(res["content"][0]["text"])
        assert payload["count"] == 1 and payload["reservations"][0].startswith("Bob | ")

    @pytest.mark.parametrize("limit", [0, 101, -5])
    def test_list_limit_bounds(self, client, limit):
        res = tool_result(call_tool(client, "list_confirmed_reservations", {"limit": limit}))
        assert res["isError"] is True

    def test_tool_schema_has_no_path_parameter(self, client):
        r = rpc(client, "tools/list", {})
        tools = {t["name"]: t for t in r.json()["result"]["tools"]}
        assert set(tools) == {"write_reservation", "list_confirmed_reservations"}
        props = set(tools["write_reservation"]["inputSchema"]["properties"])
        assert props == {
            "reservation_id", "customer_name", "car_plate",
            "start_time", "end_time", "approved_at",
        }

    def test_client_supplied_path_is_ignored_or_rejected(self, tmp_path, client):
        call_tool(client, "write_reservation", {**rec(), "file_path": str(tmp_path / "evil.txt")})
        assert not (tmp_path / "evil.txt").exists()


class TestLogging:
    def test_logs_never_contain_name_or_plate(self, client, caplog):
        with caplog.at_level(logging.DEBUG):
            call_tool(client, "write_reservation", rec(customer_name="Zelda Quux", car_plate="QQ99ZZ88"))
            call_tool(client, "write_reservation", rec(customer_name="Zelda Quux", car_plate="QQ99ZZ88"))
            call_tool(client, "write_reservation", rec(customer_name="Bad|Zelda", car_plate="QQ99ZZ88"))
        out = caplog.text
        assert "reservation_id=1" in out
        for secret in ("Zelda", "Quux", "QQ99ZZ88"):
            assert secret not in out
