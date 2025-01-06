## Contradictory, My Dear Watson - Multi-lingual Text Analysis

This project explores Natural Language Inference (NLI), a fundamental task in Natural Language Processing (NLP), focusing on determining logical relationships (entailment, contradiction, or neutrality) between pairs of sentences across multiple languages. The work was developed as part of the "Contradictory, My Dear Watson" Kaggle competition, where our ensemble approach achieved a top-tier performance, securing 6th place out of 64 teams.

## Project Objectives
Multilingual NLI Modeling: Develop robust NLI models capable of handling text in 15 languages, including English, Spanish, Hindi, and Russian.
Performance Optimization: Improve accuracy through model fine-tuning, custom architectures, and ensemble strategies.
Practical Applications: Create models applicable to real-world scenarios like chatbots, sentiment analysis, fake news detection, and fact-checking.

## Dataset
The dataset includes over 12,000 premise-hypothesis sentence pairs with balanced labels (entailment, neutral, contradiction). The data spans diverse languages, making it ideal for testing cross-lingual NLI models.

## Methodology
Data Preprocessing
Text cleaning and normalization.
Tokenization using language-specific embeddings.
Analysis of text length distributions and label balance.

## Models Explored
RoBERTa: Baseline model achieving 63.04% accuracy.
Multilingual BERT (mBERT): Improved performance with 65.72% accuracy.
XLM-RoBERTa: A cross-lingual model yielding 68.81% accuracy.
Fine-Tuned XLM-RoBERTa: Custom architecture and dropout regularization pushed accuracy to 70.26%.
Ensemble Models: Techniques like stacking, averaging, and L2 regularization achieved a peak test accuracy of 90.06%.

## Key Features
Layer-wise Fine-Tuning: Enhanced model optimization.
Ensemble Learning: Combined strengths of individual models using advanced strategies.
Custom Architectures: Designed to maximize accuracy and robustness.

## Results
Our ensemble approach with XLM-RoBERTa trained on diverse datasets (SNLI, MNLI, ANLI, XNLI) demonstrated superior performance, showcasing the value of combining multiple models for improved predictions.

## Final Metrics:
Test Accuracy: 90.06%
Ranked 6th in the Kaggle competition helded within the University of South Florida. 
Insights and Future Work
Challenges: Addressing multilingual text and handling offensive language in datasets.
Learnings: Ensemble methods significantly enhance accuracy in NLI tasks.
Recommendations: Explore advanced fine-tuning, data augmentation, and techniques for profanity handling.
