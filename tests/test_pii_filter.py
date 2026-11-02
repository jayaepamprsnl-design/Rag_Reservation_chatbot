import pytest

from app.guardrails.pii_filter import has_pii, mask_pii


def test_masks_email():
    result = mask_pii("Contact me at alice@example.com for details.")
    assert "alice@example.com" not in result
    assert "<EMAIL>" in result


def test_masks_phone():
    result = mask_pii("My number is +1 555-123-4567.")
    assert "555-123-4567" not in result


def test_masks_credit_card():
    result = mask_pii("My card is 4111 1111 1111 1111.")
    assert "4111" not in result


def test_no_pii_text_unchanged():
    text = "What are the parking rates per hour?"
    assert mask_pii(text) == text


def test_empty_string_unchanged():
    assert mask_pii("") == ""


def test_has_pii_true_for_email():
    assert has_pii("Reach me at test@domain.org") is True


def test_has_pii_false_for_generic_query():
    assert has_pii("What are the parking rates per hour?") is False
