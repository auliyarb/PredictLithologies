# PredictLithologies

This project predicts lithology codes from wireline logs and drilling data using machine learning and deep learning models.

## Dataset
- ~34,000 samples
- Features: wireline logs (GR, RHOB, NPHI, resistivity, sonic, etc.), drilling parameters
- Target: `Lithology_code` (categorical)

## Feature Engineering
- Log transformations for resistivity
- Ratios between resistivity zones
- Rolling mean and std for GR, RHOB, NPHI, ROP
- Gradients (per depth)
- Sonic & elastic features (DTC_clean, DTS, VP/VS ratio)

## Models Trained
1. **Random Forest** – baseline model
2. **XGBoost** – main model, best performer
3. **Deep Learning (1D CNN/MLP)** – sequence-aware exploration

## Model Performance
- Best model: **XGBoost**
- Metrics: Accuracy, Macro F1, Confusion Matrix
- Feature importance & SHAP plots included in notebook

## Usage
1. Open `PredictLithologies.ipynb` in Jupyter Notebook
2. Run all cells (requires packages listed below)
3. Saved the best model are located in `models/` folder:
   - `lithology_xgb_model.json`

## Requirements
```text
numpy
pandas
scikit-learn
xgboost
tensorflow
matplotlib
seaborn
