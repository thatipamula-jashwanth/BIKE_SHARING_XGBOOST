import pandas as pd
import numpy as np

#  Load already-saved train and validation targets
y_train = pd.read_csv("y_train.csv")
y_val = pd.read_csv("y_val.csv")

# Apply log-transform (safe for zeros)
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

#  Check ranges
print("Training target range (log):", y_train_log.min().values[0], "to", y_train_log.max().values[0])
print("Validation target range (log):", y_val_log.min().values[0], "to", y_val_log.max().values[0])

#  Save log-transformed targets
y_train_log.to_csv("y_train_log.csv", index=False)
y_val_log.to_csv("y_val_log.csv", index=False)
print("Log-transformed y sets saved!")
