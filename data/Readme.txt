# Multimodal Sentiment Analysis

**Group_ID**: T1_G44  
**Group_Name**: TechTitans  
**Group Leader**: Parthvi Gadekar  
**Faculty Mentor**: Mrs. Tejashree P Gurav  
**Department**: Data Science  
**Problem**: Problem 05. Multimodal Sentiment Analysis  

## Objective
Fuse GSR, EEG, facial action units (TIVA), and self-reports (PSY) to classify sentiment (positive=1, negative=2, neutral=0) using multiclass classification.

## Data Sources
- EEG_features_engineered.csv: Frequency bands, etc.
- GSR_features_engineered.csv: Skin conductance metrics.
- TIVA_features_engineered.csv: Facial action units and emotions.
- PSY_features_engineered.csv: Self-reports (ResponseTime, Result, Category as label proxy).

## Progress
- Step 1: Data preparation and synchronization via merging on Key/Participant_ID.
- Step 2: Preprocessing and feature engineering (e.g., EEG engagement index).
- Step 3.1: Baseline ML model (XGBoost with concatenated features).
- Step 4: Evaluation metrics (accuracy, macro-F1, confusion matrix).
- Step 4.2: Basic interpretability (feature importance).
- Step 4.3: Cohen's Kappa added in analysis.

## How to Run
1. Install dependencies: `pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy`.
2. Run notebooks in order: 01_preprocessing.ipynb → 02_feature_engineering.ipynb → 03_modeling_baseline_sentiment.ipynb → 06_analysis.ipynb.

## Next Steps (Post-Deadline)
- Incorporate late fusion (unimodal classifiers + stacking).
- Add multimodal transformers.
- Experiment with binary vs. 3-class classification.
- Use data augmentation (e.g., jitter for GSR).

## Notes
- Assumed Category in PSY.csv as label (mapped 1→1, 2→2, 3→0).
- Deadline: September 28, 2025.