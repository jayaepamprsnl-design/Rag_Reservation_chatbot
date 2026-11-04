"""API surface of the orchestration: debug endpoint, admin-chat path, reset, privacy."""
from __future__ import annotations

import uuid

from app.agents.admin_agent import MockAdminLLMClient
from app.api.main import app
from app.api.routes.admin import get_admin_llm


def _sid(tag):
    return f"{tag}-{uuid.uuid4().hex[:6]}"


def test_graph_state_endpoint_is_admin_protected_and_hides_personal_data(orch):
    sid = _sid("dbg")
    orch.book(sid, name="Jane Doe", plate="ABC-1234")

    assert orch.client.get(f"/graph/{sid}/state").status_code == 401
    assert orch.client.get(f"/graph/{sid}/state", headers={"X-Admin-Key": "wrong"}).status_code == 403

    r = orch.client.get(f"/graph/{sid}/state", headers=orch.headers)
    assert r.status_code == 200
    body = r.json()
    assert body["interrupted"] is True and body["booking"]["car_plate_masked"] == "ABC****34"
    assert "Jane" not in r.text and "ABC-1234" not in r.text

    unknown = orch.graph(_sid("none"))
    assert unknown["exists"] is False and unknown["next"] == []


def test_admin_chat_decision_resumes_the_graph_without_a_second_file_write(orch):
    sid = _sid("achat")
    rid = orch.book(sid)
    app.dependency_overrides[get_admin_llm] = lambda: MockAdminLLMClient(["Your booking is approved."])
    try:
        r = orch.client.post("/admin/chat", headers=orch.headers, json={"message": f"approve {rid}"})
    finally:
        app.dependency_overrides.pop(get_admin_llm, None)
    assert r.status_code == 200 and r.json()["decision_made"] is True and r.json()["file_written"] is True

    g = orch.graph(sid)
    assert g["status"] == "approved" and g["next"] == [] and g["file_written"] is True
    assert orch.send.await_count == 1          # agent wrote once; record node saw file_written and skipped


def test_reset_session_drops_the_checkpoint_and_a_later_decision_still_works(orch):
    sid = _sid("reset")
    rid = orch.book(sid)
    assert orch.client.post("/reset-session", json={"session_id": sid}).status_code == 200
    assert orch.graph(sid)["exists"] is False

    r = orch.approve(rid)                      # no thread to resume: pre-Stage-4 behaviour
    assert r.status_code == 200 and r.json()["file_written"] is True
    assert orch.send.await_count == 1


def test_checkpoint_file_never_contains_the_raw_user_message(orch):
    sid = _sid("pii")
    orch.chat(sid, "Hi, my email is jane.roe@example.com and I have a question about parking rules")
    orch.book(_sid("pii2"))
    from app.orchestration import graph as graph_mod
    graph_mod.reset_graph()                    # flush/close before reading the file
    data = (orch.tmp_path / "checkpoints.db").read_bytes()
    assert b"jane.roe@example.com" not in data
