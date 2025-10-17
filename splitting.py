import pandas as pd
from sklearn.model_selection import train_test_split

#  Load the encoded dataset
data = pd.read_csv("SeoulBikeData_encoded.csv")

#  Separate features and target
target_column = "Rented Bike Count" 
X = data.drop(columns=[target_column])
y = data[target_column]

#  Split dataset into training and validation sets
#    80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print shapes to confirm
print("Training features shape:", X_train.shape)
print("Validation features shape:", X_val.shape)
print("Training target shape:", y_train.shape)
print("Validation target shape:", y_val.shape)


# Save training set
X_train.to_csv("X_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False)

# Save validation set
X_val.to_csv("X_val.csv", index=False)
y_val.to_csv("y_val.csv", index=False)

print("Train and validation sets saved as CSV files!")