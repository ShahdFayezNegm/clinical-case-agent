"""Regression tests for checking every medication pair."""

import json

from graph_app import check_drug_interactions


def test_checks_all_pairs_for_four_medications():
    """Four medications must produce six checked pairs."""
    result = json.loads(
        check_drug_interactions.invoke(
            {
                "question": (
                    "Check warfarin, aspirin, "
                    "simvastatin and clarithromycin"
                )
            }
        )
    )

    assert result["status"] == "POTENTIAL_INTERACTION"
    assert result["interaction_found"] is True

    evidence = " ".join(result["evidence"]).lower()
    assert "warfarin" in evidence
    assert "aspirin" in evidence
    assert "simvastatin" in evidence
    assert "clarithromycin" in evidence
    assert "checked 6 medication pairs" in evidence
