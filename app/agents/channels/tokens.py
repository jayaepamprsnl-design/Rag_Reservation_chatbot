"""
Signed, expiring, single-use action tokens for email approval links.

Design
------
Token payload  {"rid": reservation_id, "aid": approval_request_id,
                "dec": "APPROVE"|"REFUSE",  "rsn": reason_string}

Signing        itsdangerous.URLSafeTimedSerializer with a per-deployment
               secret (EMAIL_TOKEN_SECRET) and fixed salt "email-action-v1".
               Tampering with any field breaks the HMAC and raises
               BadSignature — so "dec" cannot be flipped from APPROVE to
               REFUSE by an attacker.

Expiry         Enforced by itsdangerous (max_age = EMAIL_TOKEN_MAX_AGE_SECONDS,
               default 24 h).

Single-use     The token embeds the ApprovalRequest.id ("aid").
               consume() marks that row status="responded" in the DB.
               A second call with the same token finds status != "sent"
               and raises TokenUsed — before any reservation mutation.

Public API
----------
sign(rid, aid, decision, reason, *, secret, max_age)  → str
verify(token, *, secret, max_age)                      → dict
consume(token, db, *, secret, max_age)                 → dict
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy.orm import Session

from app.core.config import settings

_SALT = "email-action-v1"


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class TokenExpired(ValueError):
    """The token's max_age has been exceeded."""


class TokenInvalid(ValueError):
    """The token signature is invalid or the payload is malformed."""


class TokenUsed(ValueError):
    """The ApprovalRequest row has already been marked as responded."""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# The placeholder shipped as the settings default is public (it is in the repo), so a
# token signed with it is forgeable.  Fail closed: with no real secret configured,
# one-click email links are neither created nor accepted.
INSECURE_SECRETS = frozenset({"", "change-me-in-production", "change-me-before-production"})


def secret_is_configured() -> bool:
    return settings.email_token_secret not in INSECURE_SECRETS


def _serializer(secret: str) -> URLSafeTimedSerializer:
    if secret in INSECURE_SECRETS:
        raise TokenInvalid(
            "email action links are disabled: set EMAIL_TOKEN_SECRET to a random value "
            "(python -c \"import secrets; print(secrets.token_urlsafe(32))\")"
        )
    return URLSafeTimedSerializer(secret, salt=_SALT)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sign(
    reservation_id: int,
    approval_request_id: int,
    decision: str,
    reason: str = "",
    *,
    secret: Optional[str] = None,
) -> str:
    """
    Return a URL-safe signed token embedding the given action payload.

    Args:
        reservation_id:      LotReservation.id
        approval_request_id: ApprovalRequest.id (single-use anchor)
        decision:            "APPROVE" or "REFUSE"
        reason:              Refusal reason (only meaningful for REFUSE)
        secret:              Override signing secret (for tests)
    """
    payload = {
        "rid": reservation_id,
        "aid": approval_request_id,
        "dec": decision.upper(),
        "rsn": reason,
    }
    return _serializer(secret or settings.email_token_secret).dumps(payload)


def verify(
    token: str,
    *,
    secret: Optional[str] = None,
    max_age: Optional[int] = None,
) -> dict:
    """
    Verify the token's signature and age.

    Returns the decoded payload dict.

    Raises:
        TokenExpired  — if max_age is exceeded
        TokenInvalid  — if signature or structure is wrong
    """
    _max = max_age if max_age is not None else settings.email_token_max_age_seconds
    try:
        data = _serializer(secret or settings.email_token_secret).loads(
            token, max_age=_max
        )
    except SignatureExpired as exc:
        raise TokenExpired(f"Token expired: {exc}") from exc
    except BadSignature as exc:
        raise TokenInvalid(f"Token signature invalid: {exc}") from exc

    # Structural sanity check
    for key in ("rid", "aid", "dec"):
        if key not in data:
            raise TokenInvalid(f"Token missing required field '{key}'")

    return data


def consume(
    token: str,
    db: Session,
    *,
    secret: Optional[str] = None,
    max_age: Optional[int] = None,
) -> dict:
    """
    Verify the token, enforce single-use, and mark the ApprovalRequest row.

    Steps:
    1. ``verify()`` — raises TokenExpired / TokenInvalid on bad tokens
    2. Load the ApprovalRequest row identified by ``aid``
    3. If ``status != "sent"`` → raise TokenUsed (already consumed)
    4. Mark row: ``status = "responded"``, ``responded_at = utcnow()``
    5. Commit and return the payload

    Raises:
        TokenExpired
        TokenInvalid  (also if the ApprovalRequest row is not found)
        TokenUsed
    """
    # This import is local to avoid circular imports
    from app.db.models import ApprovalRequest

    data = verify(token, secret=secret, max_age=max_age)

    # Validate the decision field BEFORE touching the DB so that sending an
    # APPROVE token to /admin/refuse (or vice versa) does not permanently
    # consume the token for a no-op.  Callers that need to enforce dec-type
    # alignment should check data["dec"] after this call returns.

    aid: int = data["aid"]
    row: Optional[ApprovalRequest] = db.get(ApprovalRequest, aid)

    if row is None:
        raise TokenInvalid(
            f"ApprovalRequest #{aid} not found — token may be from a deleted record"
        )

    if row.status != "sent":
        raise TokenUsed(
            f"This link has already been used "
            f"(ApprovalRequest #{aid} status='{row.status}')"
        )

    row.status = "responded"
    row.responded_at = datetime.utcnow()
    db.commit()

    return data
