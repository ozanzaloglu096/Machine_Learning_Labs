import os
import hashlib
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

print("--- 1. & 2. Ingestion Step ---")
data = fetch_openml(name='adult', version=2, as_frame=True)
df = data.frame

print(f"Raw Data Loaded - Rows: {df.shape[0]}, Columns: {df.shape[1]}")
assert df.shape[0] == 48842 and df.shape[1] == 15, "Dimensions mismatch!"

target_col = df.columns[-1]
df['income'] = df[target_col].apply(lambda x: 1 if '>50K' in str(x) else 0)
if target_col != 'income':
    df.drop(columns=[target_col], inplace=True)

print("\n--- 3. Validation Step ---")
ranges = {
    'age': (17, 90),
    'hours-per-week': (1, 99),
    'capital-gain': (0, 99999),
    'capital-loss': (0, 4356)
}

violations = 0
for col, (min_v, max_v) in ranges.items():
    if col in df.columns:
        out_of_bounds = df[(df[col] < min_v) | (df[col] > max_v)].shape[0]
        print(f"Validation range check [{col}]: {out_of_bounds} violations found.")
        violations += out_of_bounds

missing_counts = (df == '?').sum().sum()
print(f"Total '?' missing values found: {missing_counts}")

print("\n--- 4. Cleaning and Transformation Step ---")
X = df.drop(columns=['income'])
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

train_df = X_train.copy()
train_df['income'] = y_train
train_df.drop_duplicates(inplace=True)
X_train = train_df.drop(columns=['income'])
y_train = train_df['income']

numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    low = X_train[col].quantile(0.01)
    high = X_train[col].quantile(0.99)
    X_train[col] = X_train[col].clip(low, high)
    X_test[col] = X_test[col].clip(low, high)

cat_cols = X_train.select_dtypes(include=['category', 'object']).columns

X_train.replace('?', np.nan, inplace=True)
X_test.replace('?', np.nan, inplace=True)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

os.makedirs('results', exist_ok=True)
joblib.dump(preprocessor, 'results/preprocessor.joblib')
print("Preprocessor saved to results/preprocessor.joblib")

print("\n--- 5. Feature Engineering Step ---")
def add_derived_features(df_in):
    df_out = df_in.copy()
    df_out['capital_net'] = df_out['capital-gain'] - df_out['capital-loss']
    df_out['hours_per_age'] = df_out['hours-per-week'] / (df_out['age'] - 16)
    return df_out

X_train_fe = add_derived_features(X_train)
X_test_fe = add_derived_features(X_test)

clf_base = LogisticRegression(max_iter=1000, random_state=42)
clf_base.fit(X_train_processed, y_train)
preds_base = clf_base.predict(X_test_processed)
print(f"Baseline (Without FE) - Accuracy: {accuracy_score(y_test, preds_base):.4f}, F1: {f1_score(y_test, preds_base):.4f}")

numeric_cols_fe = list(numeric_cols) + ['capital_net', 'hours_per_age']
preprocessor_fe = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols_fe),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ]
)

X_train_fe_proc = preprocessor_fe.fit_transform(X_train_fe)
X_test_fe_proc = preprocessor_fe.transform(X_test_fe)

clf_fe = LogisticRegression(max_iter=1000, random_state=42)
clf_fe.fit(X_train_fe_proc, y_train)
preds_fe = clf_fe.predict(X_test_fe_proc)
print(f"With Derived Features - Accuracy: {accuracy_score(y_test, preds_fe):.4f}, F1: {f1_score(y_test, preds_fe):.4f}")

print("\n--- 6. Label-Quality Check Step ---")
audit_df = X_train.copy()
audit_df['income'] = y_train
duplicates = audit_df[audit_df.duplicated(subset=list(X_train.columns), keep=False)]
conflicting_duplicates = duplicates.groupby(list(X_train.columns))['income'].nunique()
conflict_count = (conflicting_duplicates > 1).sum()
print(f"Duplicate feature rows with conflicting income labels: {conflict_count}")

print("\n--- 7. Hashing, Config & Data Card Step ---")
processed_bytes = X_train_processed.tobytes()
dataset_hash = hashlib.sha256(processed_bytes).hexdigest()
print(f"Processed Dataset SHA256 Hash: {dataset_hash}")

config_data = {
    "dataset": "UCI Adult",
    "version": 2,
    "test_size": 0.3,
    "random_state": 42,
    "dataset_sha256": dataset_hash
}
with open("results/config.yaml", "w") as f:
    yaml.dump(config_data, f)

data_card_content = f"""# Data Card: UCI Adult Dataset

- **Source**: UCI Machine Learning Repository via scikit-learn fetch_openml.
- **License**: Open Data / Public Domain.
- **Size**: 48,842 rows, 15 features.
- **Known Biases**: Historical demographic and census data from 1994.
- **Processed Dataset Hash (SHA256)**: `{dataset_hash}`
"""

with open("results/data_card.md", "w") as f:
    f.write(data_card_content)

print("Pipeline executed successfully! All artifacts stored in 'results/' directory.")
