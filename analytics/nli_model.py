"""
Classical-ML Natural Language Inference (NLI) classifier: predicts
whether a hypothesis is an entailment, neutral statement, or
contradiction relative to a premise.

This is a deliberately lightweight stand-in for the real notebook's
ensemble of fine-tuned XLM-RoBERTa transformer models (90.06% test
accuracy, 6th of 64 teams on the real Kaggle competition - see
README.md). That result required a GPU and the real ~12,000-pair
Kaggle dataset, neither available in the environment this module was
built in. This trains TF-IDF + handcrafted lexical features (word
overlap, length ratio, negation-word presence) into logistic
regression / random forest candidates, selected by real held-out
accuracy on the small hand-curated dataset in data/nli_examples.py -
useful for demonstrating the same task live and offline, not for
matching the original's accuracy.

With only ~195 hand-curated examples, the held-out accuracy reported
here is a small-sample estimate - true of any model this size, and
disclosed rather than dressed up.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data.nli_examples import LANGUAGE_NAMES, build_examples

MODEL_DIR = Path("models")
LABELS = ["entailment", "neutral", "contradiction"]

_NEGATION_WORDS = {
    "no", "not", "never", "none", "nobody", "nothing",
    "nadie", "nada", "nunca", "ningún", "ninguna", "ningun",
    "ne", "pas", "jamais", "aucun", "personne", "rien",
    "nicht", "kein", "keine", "keinen", "niemand", "nie",
    "non", "nessuno", "mai", "niente",
}


class NLIFeaturizer(BaseEstimator, TransformerMixin):
    """Turns a DataFrame with 'premise'/'hypothesis' columns into a
    feature matrix: a shared multilingual TF-IDF vocabulary applied to
    premise and hypothesis separately (so the model can see how they
    differ, not just their pooled bag-of-words), plus three handcrafted
    lexical signals long known to correlate with NLI labels: word
    overlap, length ratio, and negation-word presence differing between
    premise and hypothesis.
    """

    def __init__(self, max_features: int = 4000, ngram_range: tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range

    def fit(self, X: pd.DataFrame, y=None):
        corpus = pd.concat([X["premise"], X["hypothesis"]])
        self.vectorizer_ = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)
        self.vectorizer_.fit(corpus)
        return self

    def transform(self, X: pd.DataFrame):
        premise_vec = self.vectorizer_.transform(X["premise"])
        hypothesis_vec = self.vectorizer_.transform(X["hypothesis"])
        lexical = csr_matrix(self._lexical_features(X))
        return hstack([premise_vec, hypothesis_vec, lexical]).tocsr()

    @staticmethod
    def _lexical_features(X: pd.DataFrame) -> np.ndarray:
        rows = []
        for premise, hypothesis in zip(X["premise"], X["hypothesis"]):
            p_words = set(premise.lower().split())
            h_words = set(hypothesis.lower().split())
            overlap = len(p_words & h_words) / max(len(p_words | h_words), 1)
            len_ratio = len(h_words) / max(len(p_words), 1)
            neg_diff = int(bool(h_words & _NEGATION_WORDS)) - int(bool(p_words & _NEGATION_WORDS))
            rows.append([overlap, len_ratio, neg_diff])
        return np.array(rows, dtype=float)


CANDIDATE_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=2000),
    "random_forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42),
}


def _build_pipeline(estimator) -> Pipeline:
    return Pipeline([("features", NLIFeaturizer()), ("model", estimator)])


def load_dataset() -> pd.DataFrame:
    examples = build_examples()
    return pd.DataFrame([{"premise": e.premise, "hypothesis": e.hypothesis, "label": e.label, "language": e.language} for e in examples])


def train_and_select_best(df: pd.DataFrame | None = None, test_size: float = 0.25, seed: int = 42) -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = df if df is not None else load_dataset()

    X = df[["premise", "hypothesis"]]
    y = df["label"]
    X_train, X_test, y_train, y_test, lang_train, lang_test = train_test_split(
        X, y, df["language"], test_size=test_size, random_state=seed, stratify=y
    )

    results = {}
    best_name, best_pipeline, best_acc = None, None, -1.0
    for name, estimator in CANDIDATE_MODELS.items():
        t0 = time.time()
        pipeline = _build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = {"test_accuracy": round(acc, 4), "train_seconds": round(time.time() - t0, 2)}
        if acc > best_acc:
            best_name, best_pipeline, best_acc = name, pipeline, acc

    best_preds = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, best_preds, labels=LABELS)

    per_language_acc = {}
    for lang in sorted(set(lang_test)):
        mask = (lang_test == lang).to_numpy()
        per_language_acc[LANGUAGE_NAMES.get(lang, lang)] = round(float(accuracy_score(y_test[mask], best_preds[mask])), 4)

    feature_importance = _extract_feature_importance(best_pipeline)

    run_id = uuid.uuid4().hex[:8]
    run_record = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidates_evaluated": results,
        "selected_model": best_name,
        "test_accuracy": round(best_acc, 4),
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "baseline_accuracy": round(y_test.value_counts(normalize=True).max(), 4),  # majority-class baseline
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
        "per_language_accuracy": per_language_acc,
        "feature_importance": feature_importance,
    }

    joblib.dump(best_pipeline, MODEL_DIR / "nli_model.joblib")
    (MODEL_DIR / "latest_metrics.json").write_text(json.dumps(run_record, indent=2))
    log_path = MODEL_DIR / "run_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    log.append(run_record)
    log_path.write_text(json.dumps(log, indent=2))

    return run_record


_LEXICAL_FEATURE_NAMES = ["lexical__word_overlap", "lexical__length_ratio", "lexical__negation_word_diff"]


def _feature_names(featurizer: NLIFeaturizer) -> list[str]:
    vocab = featurizer.vectorizer_.get_feature_names_out()
    return [f"premise__{w}" for w in vocab] + [f"hypothesis__{w}" for w in vocab] + _LEXICAL_FEATURE_NAMES


def _extract_feature_importance(pipeline: Pipeline, top_n: int = 20) -> list[dict]:
    """Works for either candidate: tree-based feature_importances_, or
    logistic regression's per-class |coefficient| averaged across
    classes - surfaces which words/signals (including the 3 handcrafted
    lexical features) actually drove the selected model's predictions.
    """
    model = pipeline.named_steps["model"]
    names = _feature_names(pipeline.named_steps["features"])

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        return []

    order = np.argsort(importances)[::-1][:top_n]
    return [{"feature": names[i], "importance": round(float(importances[i]), 5)} for i in order if importances[i] > 0]


def predict(pipeline: Pipeline, premise: str, hypothesis: str) -> dict:
    X = pd.DataFrame([{"premise": premise, "hypothesis": hypothesis}])
    proba = pipeline.predict_proba(X)[0]
    classes = list(pipeline.named_steps["model"].classes_)
    return {cls: round(float(p), 4) for cls, p in zip(classes, proba)}


if __name__ == "__main__":
    record = train_and_select_best()
    print(f"Selected model: {record['selected_model']} - test accuracy: {record['test_accuracy']:.2%} (baseline: {record['baseline_accuracy']:.2%})")
    print("Per-language accuracy:", record["per_language_accuracy"])
