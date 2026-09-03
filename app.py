"""
Claim Consistency Checker - a dynamic business dashboard built on this
project's Natural Language Inference (NLI) work.

The real result this project is known for - 90.06% test accuracy, 6th
of 64 teams, from an ensemble of fine-tuned XLM-RoBERTa transformer
models trained on the actual Kaggle competition dataset - lives
untouched in NLP_Watson_FinalProject.ipynb and the accompanying report.
This dashboard is a separate, much lighter, fully offline companion:
a classical-ML NLI model (TF-IDF + handcrafted lexical features) that
can run with no GPU and no external API/network access, applied to the
practical "fact-checking" use case the original README names - given a
new claim and a set of reference statements, flag any that contradict
it. See README.md for the full picture.
"""

from __future__ import annotations

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics.consistency_checker import check_claim_against_references, summarize_consistency
from analytics.nli_model import LABELS, load_dataset, predict, train_and_select_best
from data.nli_examples import LANGUAGE_NAMES

st.set_page_config(page_title="Claim Consistency Checker", layout="wide")
st.title("Claim Consistency Checker")
st.caption(
    "A dynamic business dashboard for fact-checking: does a new claim contradict what's already on record? "
    "Built on this project's multilingual Natural Language Inference (NLI) work."
)

with st.expander("What's real here, and how does this relate to the 90.06% result?", expanded=False):
    st.markdown(
        """
This project's real, validated result - **90.06% test accuracy, 6th of 64 teams** in the
["Contradictory, My Dear Watson"](https://www.kaggle.com/competitions/contradictory-my-dear-watson) Kaggle
competition - came from an ensemble of fine-tuned XLM-RoBERTa transformer models trained on GPU against the
real ~12,000-pair, 15-language competition dataset. That work lives untouched in
[`NLP_Watson_FinalProject.ipynb`](NLP_Watson_FinalProject.ipynb) and the project report - this dashboard
doesn't reproduce or replace it.

| | The real notebook | This dashboard |
| --- | --- | --- |
| Model | Ensemble of fine-tuned XLM-RoBERTa transformers | TF-IDF + handcrafted lexical features (word overlap, length ratio, negation) into logistic regression / random forest |
| Training data | ~12,000 real pairs, 15 languages (SNLI/MNLI/ANLI/XNLI-derived) | ~195 hand-curated pairs, 5 languages - written specifically for this offline demo, not scraped |
| Compute | GPU, Google Colab | CPU, runs anywhere in seconds |
| Test accuracy | **90.06%** | Whatever it actually measures on held-out data below - typically well above the ~33% random-guess baseline for 3-class NLI, nowhere near 90% |
| Best for | The real benchmark result | Exploring the same task live, and a genuine fact-checking tool, with zero infrastructure |
        """
    )


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_dataset()


@st.cache_resource
def get_model():
    record = train_and_select_best()
    pipeline = joblib.load("models/nli_model.joblib")
    return pipeline, record


dataset = get_dataset()
pipeline, model_record = get_model()

tab_overview, tab_checker, tab_playground, tab_insights = st.tabs(
    ["Overview", "Claim Consistency Checker", "Try a Single Pair", "Model Insights"]
)

# ------------------------------------------------------------------ Overview
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training examples", len(dataset))
    c2.metric("Languages", dataset["language"].nunique())
    c3.metric("Selected model", model_record["selected_model"])
    c4.metric("Test accuracy", f"{model_record['test_accuracy']:.1%}", f"{(model_record['test_accuracy'] - model_record['baseline_accuracy']) * 100:+.1f}pp vs. baseline")

    col1, col2 = st.columns(2)
    with col1:
        label_counts = dataset["label"].value_counts()
        st.plotly_chart(px.pie(values=label_counts, names=label_counts.index, title="Dataset label balance"), use_container_width=True)
    with col2:
        lang_counts = dataset["language"].map(LANGUAGE_NAMES).value_counts()
        st.plotly_chart(px.bar(lang_counts, title="Examples per language", labels={"value": "Examples", "index": ""}), use_container_width=True)

    st.subheader("Model comparison")
    st.caption("Real candidate models, compared by actual held-out test accuracy - never hardcoded.")
    comparison = pd.DataFrame(model_record["candidates_evaluated"]).T
    comparison["selected"] = comparison.index == model_record["selected_model"]
    st.dataframe(comparison.style.format({"test_accuracy": "{:.1%}"}), use_container_width=True)

# ---------------------------------------------------- Claim Consistency Checker
with tab_checker:
    st.subheader("Check a claim against reference statements")
    st.caption(
        "Paste reference statements (e.g. policy text, prior responses, documentation) one per line, "
        "then a new claim to check against all of them."
    )
    default_refs = "The office is closed on weekends.\nEmployees must badge in before 9am on weekdays.\nRemote work is allowed up to two days per week."
    references_text = st.text_area("Reference statements (one per line)", value=default_refs, height=120)
    claim = st.text_input("New claim to check", value="The office is open every day of the week, including weekends.")

    if st.button("Check consistency", type="primary"):
        references = [r for r in references_text.splitlines() if r.strip()]
        results = check_claim_against_references(pipeline, claim, references)
        summary = summarize_consistency(results)
        st.session_state["consistency_results"] = (results, summary)

    if "consistency_results" in st.session_state:
        results, summary = st.session_state["consistency_results"]
        if summary["status"] == "CONTRADICTIONS_FOUND":
            st.error(f"⚠️ {summary['n_contradictions']} of {summary['n_references']} reference statement(s) contradict this claim.")
        elif summary["status"] == "no_references":
            st.info("Add at least one reference statement above.")
        else:
            st.success(f"✅ No contradictions detected across {summary['n_references']} reference statement(s).")

        st.dataframe(
            results.style.format({"entailment": "{:.1%}", "neutral": "{:.1%}", "contradiction": "{:.1%}"})
            .apply(lambda row: ["background-color: #ffe5e5" if row["predicted_label"] == "contradiction" else "" for _ in row], axis=1),
            use_container_width=True,
        )

# -------------------------------------------------------------- Try a Single Pair
with tab_playground:
    st.subheader("Try a single premise/hypothesis pair")
    example = dataset.sample(1, random_state=None).iloc[0]
    c1, c2 = st.columns(2)
    with c1:
        premise = st.text_area("Premise", value=example["premise"], height=100)
    with c2:
        hypothesis = st.text_area("Hypothesis", value=example["hypothesis"], height=100)

    if st.button("Classify"):
        probs = predict(pipeline, premise, hypothesis)
        predicted = max(probs, key=probs.get)
        st.markdown(f"**Prediction: `{predicted}`**")
        st.plotly_chart(
            px.bar(x=list(probs.keys()), y=list(probs.values()), labels={"x": "Label", "y": "Probability"}, title="Class probabilities"),
            use_container_width=True,
        )

# ---------------------------------------------------------------- Model Insights
with tab_insights:
    st.subheader("Confusion matrix")
    cm = model_record["confusion_matrix"]
    labels = model_record["labels"]
    fig = go.Figure(data=go.Heatmap(z=cm, x=labels, y=labels, text=cm, texttemplate="%{text}", colorscale="Blues"))
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", title=f"Confusion matrix ({model_record['selected_model']}, n={model_record['n_test_rows']} test examples)")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fi = pd.DataFrame(model_record["feature_importance"]).head(12)
        st.plotly_chart(
            px.bar(fi, x="importance", y="feature", orientation="h", title="Top feature importance").update_layout(yaxis={"categoryorder": "total ascending"}),
            use_container_width=True,
        )
        st.caption("The three handcrafted lexical features (word overlap, length ratio, negation difference) are designed to carry real NLI signal - check whether they actually rank highly above.")
    with col2:
        lang_acc = pd.Series(model_record["per_language_accuracy"])
        st.plotly_chart(px.bar(lang_acc, title="Accuracy by language (test set)", labels={"value": "Accuracy", "index": ""}), use_container_width=True)
        st.caption(f"Based on only {model_record['n_test_rows']} held-out examples total, split across 5 languages - per-language figures are small-sample estimates, not precise benchmarks.")
