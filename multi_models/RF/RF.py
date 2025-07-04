import argparse
import sys
import pickle
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def main(csv_path):
    df = pd.read_csv(csv_path)

    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc_scores = []

    correct_samples = []
    incorrect_samples = []

    for r, (train_index, test_index) in enumerate(kf.split(X, y_encoded)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        rf_classifier = RandomForestClassifier(n_estimators=500, max_depth=4, max_features="log2", random_state=42)
        rf_classifier.fit(X_train, y_train)

        y_pred = rf_classifier.predict(X_test)
        y_pred_prob = rf_classifier.predict_proba(X_test)[:, 1]

        auc_score = roc_auc_score(y_test, y_pred_prob)
        auc_scores.append(auc_score)

        print(f"Fold {r + 1} AUC Score: {auc_score:.4f}")

        correct_index = np.where(y_pred == y_test)[0]
        incorrect_index = np.where(y_pred != y_test)[0]

        correct_samples.append(df.iloc[test_index[correct_index]].assign(Prediction='Correct', Predicted_Label=y_pred[correct_index], AUC=auc_score))
        incorrect_samples.append(df.iloc[test_index[incorrect_index]].assign(Prediction='Incorrect', Predicted_Label=y_pred[incorrect_index], AUC=auc_score))

    print(f"Mean AUC Score: {np.mean(auc_scores):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RandomForest with SMOTE on input CSV.")
    parser.add_argument("-c", "--csv", required=True, help="Path to the input CSV file")

    args = parser.parse_args()
    main(args.csv)
