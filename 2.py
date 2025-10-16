# 2.py
# Fixed OneHotEncoder param compatibility (supports sklearn versions that use `sparse`
# and versions that use `sparse_output`).

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# --- Configuration ---
CSV_PATH = "loandata.csv"  # replace with path to your CSV
RANDOM_STATE = 42
TARGET_COL = "Loan_Status"


# --- Helper functions ---
def load_data(path):
    df = pd.read_csv(path)
    return df


def prepare_target(df, target_col):
    y = df[target_col].copy()
    if y.dtype == object:
        y = y.map({"Y": 1, "N": 0, "Yes": 1, "No": 0})
    y = y.fillna(0).astype(int)
    return y


def make_onehot_encoder():
    # Try the older `sparse` parameter first (scikit-learn < ~1.2),
    # otherwise fall back to `sparse_output` (scikit-learn >= ~1.2).
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_encoder())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop")

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier",
         RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"))
    ])
    return clf


def main():
    df = load_data(CSV_PATH)
    print("Data shape:", df.shape)

    df = df[df[TARGET_COL].notna()].copy()

    y = prepare_target(df, TARGET_COL)
    X = df.drop(columns=[TARGET_COL])

    numeric_features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
    categorical_features = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

    numeric_features = [c for c in numeric_features if c in X.columns]
    categorical_features = [c for c in categorical_features if c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)
    print("Model trained.")

    y_pred = pipeline.predict(X_test)
    y_proba = None
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC: {auc:.4f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if y_proba is not None:
        cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        print(f"Cross-val ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance extraction (works for RandomForest)
    clf = pipeline.named_steps["classifier"]

    # Safely get one-hot feature names across sklearn versions
    try:
        ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
        if hasattr(ohe, "get_feature_names_out"):
            cat_cols = list(ohe.get_feature_names_out(categorical_features))
        elif hasattr(ohe, "get_feature_names"):
            # older sklearn
            cat_cols = list(ohe.get_feature_names(categorical_features))
        else:
            # fallback: create naive feature names from unique values
            cat_cols = []
            for col in categorical_features:
                unique_vals = X[col].dropna().unique()
                cat_cols.extend([f"{col}_{v}" for v in unique_vals])
    except Exception:
        cat_cols = []
        for col in categorical_features:
            unique_vals = X[col].dropna().unique()
            cat_cols.extend([f"{col}_{v}" for v in unique_vals])

    feature_names = numeric_features + cat_cols
    importances = clf.feature_importances_
    fi = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print("\nTop feature importances:")
    for name, imp in fi[:10]:
        print(f"{name}: {imp:.4f}")

    joblib.dump(pipeline, "loan_approval_pipeline.joblib")
    print("Saved pipeline to loan_approval_pipeline.joblib")

    example = X_test.iloc[[0]]
    pred = pipeline.predict(example)
    proba = None
    try:
        proba = pipeline.predict_proba(example)[:, 1]
    except Exception:
        proba = None
    print("\nExample prediction (first test row):")
    print("Predicted label:", pred[0])
    if proba is not None:
        print("Predicted probability:", float(proba[0]))


if __name__ == "__main__":
    main()