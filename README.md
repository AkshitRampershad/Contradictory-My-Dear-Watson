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

## Repository Contents
| File | Description |
| --- | --- |
| [`NLP_Watson_FinalProject.ipynb`](NLP_Watson_FinalProject.ipynb) | Full notebook: preprocessing, baseline models, fine-tuning, and ensembling |
| [`Contradictory, My Dear Watson_Final_Report_Group5.pdf`](Contradictory,%20My%20Dear%20Watson_Final_Report_Group5.pdf) | Written project report |

## Running the Notebook
The notebook was built for Google Colab and expects the Kaggle competition's `train.csv`/`test.csv` to be uploaded at runtime (via `google.colab.files.upload()`). To run it:
1. Open `NLP_Watson_FinalProject.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Download the train/test CSVs from the [Kaggle competition data page](https://www.kaggle.com/competitions/contradictory-my-dear-watson/data).
3. Run the notebook cells in order, uploading the CSVs when prompted.

Core dependencies: `tensorflow`, `transformers`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`.

## License
Released under the [MIT License](LICENSE).
