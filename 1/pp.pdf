# preprocess_and_model.py
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("placement_data.csv")

# Handle missing values
imputer = SimpleImputer(strategy="mean")
numeric_cols = ['Tenth %', 'Twelfth %', 'FE %', 'SE %', 'TE %']
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Drop duplicate rows
df = df.drop_duplicates()

# Handle outliers using IQR method
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Encode Placed column
df['Placed'] = df['Placed'].map({'Yes': 1, 'No': 0})

# Apply scaling (Standardization)
features = ['Tenth %', 'Twelfth %', 'FE %', 'SE %', 'TE %', 'Certifications', 'Projects', 'Internships']
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Split data
X = df[features]
y = df['Placed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy check
preds = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, preds))

# Save model
pickle.dump(model, open("placement_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
