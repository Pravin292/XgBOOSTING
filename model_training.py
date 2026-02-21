import pandas as pd
import numpy as np
import joblib
import json
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import fetch_california_housing # using cali for standard xgboost example regression -> classification adaptation if we want or just raw xgboost examples. Wait let's fetch a classification one

# We'll use the breast cancer dataset natively but mapped through xgboost specifically for this directory logic as a reliable classification baseline
from sklearn.datasets import load_diabetes

def train_xgboost_model():
    print("Fetching diabetes dataset from sklearn for XGBoost Regression...")
    try:
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/diabetes.csv", index=False)
        print("Dataset saved to data/diabetes.csv")
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        return

    # Features and Target
    X = df.drop(columns=['target'])
    y = df['target'] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 100)
    model.fit(X_train, y_train)

    print("Evaluating Model...")
    from sklearn.metrics import mean_squared_error, r2_score
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    features_list = list(X.columns)
    
    metrics = {
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Model": "XGBoost Regressor"
    }

    # Feature Importance for Explanation
    feature_importance = dict(zip(X.columns, np.round(model.feature_importances_, 4)))
    # convert float32 to float for json serialization
    feature_importance = {k: float(v) for k, v in feature_importance.items()}
    metrics["feature_importance"] = feature_importance
    
    # Dump files required for the application
    joblib.dump(model, "xgb_model.pkl")
    joblib.dump(features_list, "feature_columns.pkl")
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\nTraining complete. Model and metrics saved.")

if __name__ == "__main__":
    train_xgboost_model()
