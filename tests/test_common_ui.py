"""
Tests for app/ui/common.py — API helper functions and error display.

Uses unittest.mock to intercept httpx calls; never touches the real backend.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.ui.common import (
    API_URL,
    _ADMIN_KEY_HEADER,
    api_get,
    api_post,
    status_badge,
)


# ---------------------------------------------------------------------------
# api_get
# ---------------------------------------------------------------------------


class TestApiGet:

    def test_success_returns_status_and_json(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"id": 1}]

        with patch("app.ui.common.httpx.get", return_value=mock_resp) as m:
            code, data = api_get("/admin/reservations/pending")

        assert code == 200
        assert data == [{"id": 1}]
        m.assert_called_once()
        call_url = m.call_args[0][0]
        assert call_url == f"{API_URL}/admin/reservations/pending"

    def test_connect_error_returns_zero(self):
        import httpx
        with patch("app.ui.common.httpx.get", side_effect=httpx.ConnectError("refused")):
            code, data = api_get("/health")
        assert code == 0
        assert "detail" in data

    def test_timeout_returns_minus_one(self):
        import httpx
        with patch("app.ui.common.httpx.get", side_effect=httpx.TimeoutException("")):
            code, data = api_get("/health")
        assert code == -1
        assert "detail" in data

    def test_headers_forwarded(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        with patch("app.ui.common.httpx.get", return_value=mock_resp) as m:
            api_get("/admin/verify", headers={"X-Admin-Key": "mykey"})
        _, kwargs = m.call_args
        assert kwargs.get("headers", {}).get("X-Admin-Key") == "mykey"

    def test_query_params_forwarded(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        with patch("app.ui.common.httpx.get", return_value=mock_resp) as m:
            api_get("/admin/reservations/history", limit=50)
        _, kwargs = m.call_args
        assert kwargs.get("params", {}).get("limit") == 50

    def test_non_json_exception_returns_minus_two(self):
        with patch("app.ui.common.httpx.get", side_effect=ValueError("oops")):
            code, data = api_get("/health")
        assert code == -2


# ---------------------------------------------------------------------------
# api_post
# ---------------------------------------------------------------------------


class TestApiPost:

    def test_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok"}

        with patch("app.ui.common.httpx.post", return_value=mock_resp):
            code, data = api_post("/admin/reservations/1/approve", json={"note": ""})

        assert code == 200
        assert data["status"] == "ok"

    def test_connect_error(self):
        import httpx
        with patch("app.ui.common.httpx.post", side_effect=httpx.ConnectError("x")):
            code, data = api_post("/admin/chat", json={"message": "approve 1"})
        assert code == 0

    def test_timeout(self):
        import httpx
        with patch("app.ui.common.httpx.post", side_effect=httpx.TimeoutException("")):
            code, data = api_post("/admin/chat", json={})
        assert code == -1


# ---------------------------------------------------------------------------
# status_badge
# ---------------------------------------------------------------------------


class TestStatusBadge:

    @pytest.mark.parametrize("status,expected_emoji", [
        ("pending",   "🟡"),
        ("confirmed", "🟢"),
        ("refused",   "🔴"),
        ("expired",   "⬜"),
        ("cancelled", "⬛"),
    ])
    def test_known_status(self, status, expected_emoji):
        assert expected_emoji in status_badge(status)

    def test_unknown_status_uppercased(self):
        assert status_badge("weirdstate") == "WEIRDSTATE"

    def test_case_insensitive(self):
        assert status_badge("CONFIRMED") == status_badge("confirmed")
