# 🩺 XGBOOSTING: Diabetes Progression (Midnight Glass Edition)

A high-performance **XGBoost Regression** engine for quantitative disease evaluation.

## 🏗 Project Structure
- `app.py`: Streamlit UI handling normalized parameter input and regression visuals.
- `model_training.py`: Full training pipeline for the XGBoost Regressor.
- `utils.py`: Visualization and loading helper modules.
- `requirements.txt`: Multi-project isolated dependencies.
- `data/`: Local source Informatics data.

## 🚀 Usage Instructions

### 1. Training Pipeline
Execute the XGBoost engine setup:
```bash
python3 model_training.py
```

### 2. Predictive Console
Launch the progression analytics app:
```bash
streamlit run app.py
```

## 📊 Analytics
- Metric Focus: **MSE, RMSE, and R2 Correlation**.
- Advanced Logic: Uses XGBoost's `reg:squarederror` objective.
- Responsive UX: Normalized sidebar inputs for precise parameter control.