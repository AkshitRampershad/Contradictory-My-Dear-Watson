# Contradictory, My Dear Watson — Multi-lingual Text Analysis

This project explores Natural Language Inference (NLI), a fundamental NLP task focused on determining the logical relationship — entailment, contradiction, or neutrality — between pairs of sentences across multiple languages. It was developed for the ["Contradictory, My Dear Watson"](https://www.kaggle.com/competitions/contradictory-my-dear-watson) Kaggle competition (held within the University of South Florida), where our ensemble approach placed **6th out of 64 teams** with a **90.06% test accuracy**.

## Project Objectives
- **Multilingual NLI Modeling**: Develop robust NLI models capable of handling text in 15 languages, including English, Spanish, Hindi, and Russian.
- **Performance Optimization**: Improve accuracy through model fine-tuning, custom architectures, and ensemble strategies.
- **Practical Applications**: Build models applicable to real-world scenarios like chatbots, sentiment analysis, fake news detection, and fact-checking.

## Dataset
Over 12,000 premise-hypothesis sentence pairs with balanced labels (entailment, neutral, contradiction), spanning 15 languages — well suited for testing cross-lingual NLI models.

## Methodology
- **Data preprocessing**: text cleaning and normalization, tokenization using language-specific embeddings, analysis of text length distributions and label balance.
- **Model progression**: baseline transformer models, evaluated and iterated on, culminating in an ensemble of fine-tuned models.
- **Ensemble learning**: stacking, averaging, and L2 regularization across multiple XLM-RoBERTa models (including a bidirectional-LSTM variant) trained on SNLI, MNLI, ANLI, and XNLI.

## Models Explored
| Model | Test Accuracy |
| --- | --- |
| RoBERTa (baseline) | 63.04% |
| Multilingual BERT (mBERT) | 65.72% |
| XLM-RoBERTa | 68.81% |
| Fine-tuned XLM-RoBERTa (custom architecture + dropout regularization) | 70.26% |
| **Ensemble** (stacking + averaging + L2 regularization) | **90.06%** |

## Key Features
- **Layer-wise fine-tuning** for enhanced model optimization
- **Ensemble learning** combining the strengths of individual models via stacking and averaging
- **Custom architectures** designed to maximize accuracy and robustness across languages

## Results
The final ensemble of XLM-RoBERTa models, trained across SNLI, MNLI, ANLI, and XNLI datasets, delivered the strongest performance — demonstrating that combining multiple fine-tuned models substantially outperforms any single model on cross-lingual NLI.

- **Test Accuracy**: 90.06%
- **Ranking**: 6th of 64 teams in the Kaggle competition

## Insights & Future Work
- **Challenges**: handling multilingual text consistently and filtering offensive language in the datasets.
- **Learnings**: ensemble methods significantly enhance accuracy on NLI tasks.
- **Recommendations**: explore further fine-tuning strategies, data augmentation, and dedicated profanity handling.

## Live Dashboard: Claim Consistency Checker

Alongside the notebook above, this repo also includes a **dynamic business dashboard** built on the same NLI task: a fact-checking tool that checks whether a new claim contradicts a set of reference statements (e.g. policy text, prior responses, documentation) — directly matching this project's stated "fact-checking" practical application.

**Important distinction:** the dashboard does **not** reproduce the 90.06% ensemble above. That result required a GPU and the real ~12,000-pair Kaggle dataset. The dashboard was built in an environment with neither (no GPU, no network access to Kaggle or HuggingFace), so it uses a much lighter, fully offline model instead:

| | The notebook above | The dashboard (`app.py`) |
| --- | --- | --- |
| Model | Ensemble of fine-tuned XLM-RoBERTa transformers | TF-IDF + handcrafted lexical features (word overlap, length ratio, negation) into logistic regression / random forest |
| Training data | ~12,000 real pairs, 15 languages (SNLI/MNLI/ANLI/XNLI-derived) | ~195 hand-curated pairs, 5 languages, written specifically for this offline demo (see `data/nli_examples.py`) — not scraped or a subset of the real Kaggle data |
| Compute | GPU, Google Colab | CPU, runs anywhere in seconds |
| Test accuracy | **90.06%** | Whatever it actually measures on held-out data — comfortably above the ~33% random-guess baseline for 3-class NLI, nowhere near 90.06% |
| Purpose | The real competition benchmark | An interactive, zero-infrastructure companion demonstrating the same task live |

### What's in the dashboard
- **Overview** — dataset composition, and a real comparison of candidate models by held-out test accuracy.
- **Claim Consistency Checker** — the main tool: paste reference statements and a new claim, get every reference scored against it with any contradictions flagged.
- **Try a Single Pair** — a simple premise/hypothesis playground.
- **Model Insights** — confusion matrix, feature importance (do the handcrafted lexical features actually rank highly? — check for yourself), and per-language accuracy, with an explicit small-sample caveat given the dataset's size.

## Repository Contents
| File | Description |
| --- | --- |
| [`NLP_Watson_FinalProject.ipynb`](NLP_Watson_FinalProject.ipynb) | Full notebook: preprocessing, baseline models, fine-tuning, and ensembling (the real 90.06% result) |
| [`Contradictory, My Dear Watson_Final_Report_Group5.pdf`](Contradictory,%20My%20Dear%20Watson_Final_Report_Group5.pdf) | Written project report |
| `app.py` | The Claim Consistency Checker dashboard |
| `data/nli_examples.py` | Hand-curated multilingual NLI dataset for the dashboard |
| `analytics/nli_model.py` | Trains and selects the classical-ML NLI classifier |
| `analytics/consistency_checker.py` | Checks a claim against reference statements, flags contradictions |
| `tests/` | Pytest suite for the dataset, model, and consistency checker |

## Running the Notebook
The notebook was built for Google Colab and expects the Kaggle competition's `train.csv`/`test.csv` to be uploaded at runtime (via `google.colab.files.upload()`). To run it:
1. Open `NLP_Watson_FinalProject.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Download the train/test CSVs from the [Kaggle competition data page](https://www.kaggle.com/competitions/contradictory-my-dear-watson/data).
3. Run the notebook cells in order, uploading the CSVs when prompted.

Core dependencies: `tensorflow`, `transformers`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`.

## License
Released under the [MIT License](LICENSE).
