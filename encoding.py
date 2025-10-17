import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load dataset with proper encoding
data = pd.read_csv("SeoulBikeData.csv", encoding='latin1')  # fix UnicodeDecodeError

#  Inspect data
print(data.head())
print(data.info())

#  Process Date column: extract Month, Day, Weekday (optional, better than one-hot full date)
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Weekday'] = data['Date'].dt.weekday
# Drop original Date column
data = data.drop(columns=['Date'])

# Identify categorical columns automatically
categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
print("Categorical Columns to encode:", categorical_cols)

#  One-hot encoding using sklearn (for sklearn >=1.2)
ohe = OneHotEncoder(sparse_output=False, drop='first')

encoded_data = pd.DataFrame(
    ohe.fit_transform(data[categorical_cols]),
    columns=ohe.get_feature_names_out(categorical_cols)
)

#  Drop original categorical columns and merge encoded columns
data = data.drop(columns=categorical_cols)
data = pd.concat([data, encoded_data], axis=1)

#  Save the encoded dataset to a new CSV
data.to_csv("SeoulBikeData_encoded.csv", index=False)
print("Encoded dataset saved as 'SeoulBikeData_encoded.csv'")

#  Optional: check final dataset
print(data.head())
print(data.info())
