import joblib
import pytest

from analytics.consistency_checker import check_claim_against_references, summarize_consistency
from analytics.nli_model import train_and_select_best


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    import os

    workdir = tmp_path_factory.mktemp("model")
    old_cwd = os.getcwd()
    os.chdir(workdir)
    train_and_select_best(seed=42)
    pipeline = joblib.load("models/nli_model.joblib")
    os.chdir(old_cwd)
    return pipeline


def test_empty_references_returns_empty_result(pipeline):
    result = check_claim_against_references(pipeline, "Some claim.", [])
    assert result.empty
    summary = summarize_consistency(result)
    assert summary["status"] == "no_references"


def test_results_sorted_by_contradiction_probability_descending(pipeline):
    result = check_claim_against_references(
        pipeline,
        "The office is open every day of the week, including weekends.",
        ["The office is closed on weekends.", "The office has a rooftop garden.", "Employees badge in at 9am."],
    )
    probs = result["contradiction"].tolist()
    assert probs == sorted(probs, reverse=True)


def test_summary_status_matches_predicted_labels_exactly(pipeline):
    """Regression test: summarize_consistency used to gate on a
    separate confidence threshold that could disagree with the model's
    own top prediction, producing a status that contradicted
    n_contradictions > 0. status must always match predicted_label.
    """
    result = check_claim_against_references(
        pipeline,
        "The office is open every day of the week, including weekends.",
        ["The office is closed on weekends.", "Employees badge in at 9am.", "There is a garden."],
    )
    summary = summarize_consistency(result)
    n_label_contradictions = int((result["predicted_label"] == "contradiction").sum())
    assert summary["n_contradictions"] == n_label_contradictions
    if n_label_contradictions > 0:
        assert summary["status"] == "CONTRADICTIONS_FOUND"
    else:
        assert summary["status"] == "NO_CONTRADICTIONS_DETECTED"


def test_flagged_entries_carry_confidence(pipeline):
    result = check_claim_against_references(
        pipeline,
        "The office is open every day of the week, including weekends.",
        ["The office is closed on weekends."],
    )
    summary = summarize_consistency(result)
    if summary["flagged"]:
        entry = summary["flagged"][0]
        assert "reference" in entry and "confidence" in entry
        assert 0 <= entry["confidence"] <= 1


def test_whitespace_only_references_are_ignored(pipeline):
    result = check_claim_against_references(pipeline, "A claim.", ["", "   ", "A real reference."])
    assert len(result) == 1
