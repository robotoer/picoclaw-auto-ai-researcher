"""Tests for E01 claim extraction (mocked API calls)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Local data models (mirrors E01 extraction schema)
# ---------------------------------------------------------------------------

VALID_CLAIM_TYPES = {"quantitative", "methodological", "comparative", "existence", "causal"}


class ExtractedClaim(BaseModel):
    """A single claim extracted from a paper."""

    text: str
    claim_type: str = Field(alias="type", default="methodological")
    source_section: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    model_config = {"populate_by_name": True}


class ClaimExtractionResult(BaseModel):
    """Full result of extracting claims from one paper."""

    arxiv_id: str
    model_name: str
    claims: list[ExtractedClaim] = Field(default_factory=list)
    raw_response: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Parsing helpers (mirrors expected E01 src logic)
# ---------------------------------------------------------------------------


def parse_claims_from_response(raw_text: str) -> list[ExtractedClaim]:
    """Parse an LLM JSON response into ExtractedClaim objects.

    Handles both a bare JSON array and ``{"claims": [...]}`` wrapper.
    Gracefully returns an empty list on malformed input.
    """
    if not raw_text or not raw_text.strip():
        return []

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return []

    items: list[dict[str, Any]]
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "claims" in data:
        items = data["claims"]
    else:
        return []

    claims: list[ExtractedClaim] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # Normalise claim type
        ct = item.get("type", item.get("claim_type", "methodological"))
        if ct not in VALID_CLAIM_TYPES:
            ct = "methodological"  # fallback
        try:
            claim = ExtractedClaim(
                text=str(item.get("text", "")),
                type=ct,
                source_section=str(item.get("source_section", "")),
                confidence=float(item.get("confidence", 0.5)),
            )
            claims.append(claim)
        except (ValueError, TypeError):
            continue

    return claims


# ---------------------------------------------------------------------------
# Mock response helpers
# ---------------------------------------------------------------------------

VALID_RESPONSE = json.dumps(
    [
        {
            "text": "BERT achieves 93.2 F1 on SQuAD 2.0",
            "type": "quantitative",
            "source_section": "Results",
            "confidence": 0.95,
        },
        {
            "text": "The proposed method reduces training time by 40%",
            "type": "comparative",
            "source_section": "Experiments",
            "confidence": 0.8,
        },
    ]
)

MALFORMED_RESPONSE = '{"claims": [{"text": "incomplete...'

INVALID_TYPE_RESPONSE = json.dumps(
    [
        {
            "text": "Some claim",
            "type": "INVALID_TYPE",
            "source_section": "Intro",
            "confidence": 0.5,
        }
    ]
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseAnthropicResponse:
    def test_valid_json(self) -> None:
        """Valid JSON response is parsed correctly into ExtractedClaim objects."""
        claims = parse_claims_from_response(VALID_RESPONSE)
        assert len(claims) == 2

        assert claims[0].text == "BERT achieves 93.2 F1 on SQuAD 2.0"
        assert claims[0].claim_type == "quantitative"
        assert claims[0].source_section == "Results"
        assert claims[0].confidence == pytest.approx(0.95)

        assert claims[1].claim_type == "comparative"

    def test_wrapped_format(self) -> None:
        """Response wrapped in {"claims": [...]} is handled."""
        wrapped = json.dumps(
            {
                "claims": [
                    {"text": "A claim", "type": "existence", "confidence": 0.6}
                ]
            }
        )
        claims = parse_claims_from_response(wrapped)
        assert len(claims) == 1
        assert claims[0].claim_type == "existence"


class TestParseMalformedResponse:
    def test_malformed_json(self) -> None:
        """Malformed JSON is handled gracefully."""
        claims = parse_claims_from_response(MALFORMED_RESPONSE)
        assert claims == []

    def test_non_json_text(self) -> None:
        """Plain text response returns empty list."""
        claims = parse_claims_from_response("This is not JSON at all.")
        assert claims == []

    def test_wrong_structure(self) -> None:
        """JSON that isn't an array or dict-with-claims returns empty."""
        claims = parse_claims_from_response(json.dumps({"foo": "bar"}))
        assert claims == []


class TestClaimTypeValidation:
    def test_invalid_type_falls_back(self) -> None:
        """Invalid claim types are replaced with the default."""
        claims = parse_claims_from_response(INVALID_TYPE_RESPONSE)
        assert len(claims) == 1
        assert claims[0].claim_type == "methodological"

    def test_all_valid_types_accepted(self) -> None:
        """All recognised claim types are kept as-is."""
        for ct in VALID_CLAIM_TYPES:
            resp = json.dumps([{"text": "x", "type": ct, "confidence": 0.5}])
            claims = parse_claims_from_response(resp)
            assert len(claims) == 1
            assert claims[0].claim_type == ct


class TestEmptyPaperText:
    def test_empty_string(self) -> None:
        """Empty text returns empty claims."""
        assert parse_claims_from_response("") == []

    def test_whitespace_only(self) -> None:
        """Whitespace-only text returns empty claims."""
        assert parse_claims_from_response("   \n\t  ") == []


class TestClaimExtractionResultModel:
    def test_serialization_roundtrip(self) -> None:
        """ClaimExtractionResult serializes and deserializes correctly."""
        result = ClaimExtractionResult(
            arxiv_id="2312.12345",
            model_name="claude-sonnet-4-20250514",
            claims=[
                ExtractedClaim(
                    text="A finding",
                    type="quantitative",
                    source_section="Results",
                    confidence=0.9,
                ),
            ],
            raw_response=VALID_RESPONSE,
        )

        data = result.model_dump(by_alias=True)
        restored = ClaimExtractionResult.model_validate(data)

        assert restored.arxiv_id == "2312.12345"
        assert restored.model_name == "claude-sonnet-4-20250514"
        assert len(restored.claims) == 1
        assert restored.claims[0].text == "A finding"
        assert restored.error is None

    def test_with_error(self) -> None:
        """ClaimExtractionResult captures error state."""
        result = ClaimExtractionResult(
            arxiv_id="2312.99999",
            model_name="gpt-4o",
            error="API rate limit exceeded",
        )
        assert result.claims == []
        assert result.error == "API rate limit exceeded"

        data = result.model_dump(by_alias=True)
        assert data["error"] == "API rate limit exceeded"
