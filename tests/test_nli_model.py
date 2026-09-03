import json
from pathlib import Path

from analytics.nli_model import LABELS, load_dataset, predict, train_and_select_best
from data.nli_examples import LANGUAGE_NAMES


def test_model_beats_majority_baseline():
    record = train_and_select_best(seed=42)
    assert record["test_accuracy"] > record["baseline_accuracy"], (
        f"Model ({record['test_accuracy']}) should beat the majority-class baseline ({record['baseline_accuracy']})"
    )


def test_model_beats_baseline_across_multiple_seeds():
    """A single lucky split isn't enough evidence with a dataset this
    small - check several splits."""
    wins = 0
    for seed in [1, 2, 3, 4, 5]:
        record = train_and_select_best(seed=seed)
        if record["test_accuracy"] > record["baseline_accuracy"]:
            wins += 1
    assert wins >= 4, f"Model beat baseline in only {wins}/5 seeds"


def test_confusion_matrix_shape():
    record = train_and_select_best(seed=42)
    cm = record["confusion_matrix"]
    assert len(cm) == len(LABELS)
    assert all(len(row) == len(LABELS) for row in cm)
    assert sum(sum(row) for row in cm) == record["n_test_rows"]


def test_feature_importance_includes_lexical_features():
    record = train_and_select_best(seed=42)
    assert record["feature_importance"]
    feature_names = {f["feature"] for f in record["feature_importance"]}
    assert any(name.startswith("lexical__") for name in feature_names), (
        f"Expected at least one handcrafted lexical feature in top features, got {feature_names}"
    )


def test_per_language_accuracy_uses_real_language_names():
    record = train_and_select_best(seed=42)
    assert record["per_language_accuracy"]
    assert set(record["per_language_accuracy"]).issubset(set(LANGUAGE_NAMES.values()))


def test_predict_returns_valid_probability_distribution():
    import joblib

    train_and_select_best(seed=42)
    pipeline = joblib.load("models/nli_model.joblib")
    probs = predict(pipeline, "A man is playing guitar on a stage.", "A person is performing music.")
    assert set(probs) == set(LABELS)
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert all(0 <= p <= 1 for p in probs.values())


def test_run_log_accumulates_across_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_and_select_best(seed=1)
    train_and_select_best(seed=2)
    log = json.loads(Path("models/run_log.json").read_text())
    assert len(log) == 2


def test_load_dataset_matches_build_examples():
    df = load_dataset()
    assert set(df.columns) == {"premise", "hypothesis", "label", "language"}
    assert len(df) >= 150
