"""
Claim-consistency checker: the business-facing use case this project's
README names explicitly ("fact-checking"). Given a new claim and a set
of reference statements (e.g. a policy document, prior support
responses, a knowledge base), checks the claim against every reference
through the trained NLI model and flags any predicted contradictions.

Each reference is treated as the NLI "premise" and the new claim as the
"hypothesis" - i.e. the question asked per reference is "does this
reference statement entail, sit neutrally alongside, or contradict the
new claim?"
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline


def check_claim_against_references(pipeline: Pipeline, claim: str, references: list[str]) -> pd.DataFrame:
    references = [r.strip() for r in references if r.strip()]
    if not references:
        return pd.DataFrame(columns=["reference", "predicted_label", "entailment", "neutral", "contradiction"])

    X = pd.DataFrame({"premise": references, "hypothesis": [claim] * len(references)})
    proba = pipeline.predict_proba(X)
    classes = list(pipeline.named_steps["model"].classes_)
    preds = pipeline.predict(X)

    rows = []
    for reference, pred, probs in zip(references, preds, proba):
        row = {"reference": reference, "predicted_label": pred}
        row.update({cls: round(float(p), 4) for cls, p in zip(classes, probs)})
        rows.append(row)

    result = pd.DataFrame(rows)
    return result.sort_values("contradiction", ascending=False).reset_index(drop=True)


def summarize_consistency(results: pd.DataFrame) -> dict:
    """A simple, transparent summary keyed on the model's own top
    prediction per reference (not a second, separately-tuned confidence
    threshold that could silently disagree with it and produce a
    confusing "predicted contradiction, but not flagged" mismatch).
    Each flagged reference carries its predicted-class confidence so
    the user can judge how sure the model is, without that number
    gating whether it's flagged at all.
    """
    if results.empty:
        return {"status": "no_references", "n_references": 0, "n_contradictions": 0, "flagged": []}

    flagged = results[results["predicted_label"] == "contradiction"]
    status = "CONTRADICTIONS_FOUND" if len(flagged) > 0 else "NO_CONTRADICTIONS_DETECTED"
    return {
        "status": status,
        "n_references": len(results),
        "n_contradictions": len(flagged),
        "flagged": [
            {"reference": row["reference"], "confidence": row["contradiction"]}
            for _, row in flagged.iterrows()
        ],
    }


if __name__ == "__main__":
    import joblib

    from analytics.nli_model import train_and_select_best

    train_and_select_best()
    pipeline = joblib.load("models/nli_model.joblib")

    claim = "The office is open every day of the week, including weekends."
    references = [
        "The office is closed on weekends.",
        "Employees must badge in before 9am on weekdays.",
        "The office has a rooftop garden.",
    ]
    results = check_claim_against_references(pipeline, claim, references)
    print(results.to_string(index=False))
    print()
    print(summarize_consistency(results))
