import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

#  Load datasets
X_train = pd.read_csv("X_train.csv")
X_val = pd.read_csv("X_val.csv")
y_train_log = pd.read_csv("y_train_log.csv").iloc[:, 0]
y_val_log = pd.read_csv("y_val_log.csv").iloc[:, 0]

#  Load trained XGBoost model from .pkl
with open("xgboost_bike_model.pkl", "rb") as f:
    model = pickle.load(f)

#  Create DMatrix for predictions
dtrain = xgb.DMatrix(X_train)
dval = xgb.DMatrix(X_val)

#  Predict log-transformed values
y_train_pred_log = model.predict(dtrain)
y_val_pred_log = model.predict(dval)

#  Inverse log-transform to get original bike counts
y_train_pred = np.expm1(y_train_pred_log)
y_val_pred = np.expm1(y_val_pred_log)
y_train_actual = np.expm1(y_train_log)
y_val_actual = np.expm1(y_val_log)

#  Compute metrics
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

train_mse, train_rmse, train_mae, train_r2 = regression_metrics(y_train_actual, y_train_pred)
val_mse, val_rmse, val_mae, val_r2 = regression_metrics(y_val_actual, y_val_pred)

#  Print results
print("=== Training Metrics ===")
print(f"MSE: {train_mse:.2f}, RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.4f}\n")

print("=== Validation Metrics ===")
print(f"MSE: {val_mse:.2f}, RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.4f}")
