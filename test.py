import pandas as pd
import xgboost as xgb
import numpy as np
import pickle

#  Load validation set
X_val = pd.read_csv("X_val.csv")
y_val_log = pd.read_csv("y_val_log.csv").iloc[:, 0]

#  Load trained XGBoost model
with open("xgboost_bike_model_tuned.pkl", "rb") as f:
    model = pickle.load(f)

# Pick a single row to test )
row_index = 222
X_test_row = X_val.iloc[[row_index]] 
y_actual_log = y_val_log.iloc[row_index]

#  Predict using model
dtest = xgb.DMatrix(X_test_row)
y_pred_log = model.predict(dtest)[0]

#  Invert log-transform
y_pred = np.expm1(y_pred_log)
y_actual = np.expm1(y_actual_log)

#  Print results
print(f"Actual bike count: {y_actual}")
print(f"Predicted bike count: {y_pred:.2f}")
