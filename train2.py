import pandas as pd
import xgboost as xgb
import numpy as np
import pickle

#  Load datasets
X_train = pd.read_csv("X_train.csv")
X_val = pd.read_csv("X_val.csv")
y_train_log = pd.read_csv("y_train_log.csv").iloc[:, 0]
y_val_log = pd.read_csv("y_val_log.csv").iloc[:, 0]

#  Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train_log)
dval = xgb.DMatrix(X_val, label=y_val_log)

#  Updated parameters to reduce overfitting
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.1,
    "max_depth": 4,            
    "subsample": 0.9,          
    "colsample_bytree": 0.8,  
    "alpha": 1.0,              
    "lambda": 2.0,             
    "seed": 42
}

#  Train with early stopping
evallist = [(dtrain, "train"), (dval, "eval")]
num_round = 500  # reduced number of trees

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_round,
    evals=evallist,
    early_stopping_rounds=50,
    verbose_eval=50
)

# 5️⃣ Save model as .pkl
with open("xgboost_bike_model_tuned.pkl", "wb") as f:
    pickle.dump(model, f)

print("Tuned model trained and saved as 'xgboost_bike_model_tuned.pkl'")
