# src/model_training.py

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use("Agg")  # prevents chart popup issues on Mac
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def load_features(filename="features.csv"):
    path = os.path.join("data", "processed", filename)
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"Loaded {len(df)} rows for training.")
    return df


def split_data(df):
    feature_columns = [
        "Open", "High", "Low", "Close", "Volume",
        "MA20", "MA50", "Daily_Return", "Volatility", "Price_Range"
    ]

    X = df[feature_columns]
    y = df["Target"]

    print(f"\nFull dataset target distribution:")
    print(f"  BUY  (1): {(y==1).sum()}")
    print(f"  SELL (0): {(y==0).sum()}")

    # shuffle=True fixes the imbalance in test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=True,          # ← key fix
        stratify=y             # ← ensures equal BUY/SELL ratio in both sets
    )

    print(f"\nTraining rows : {len(X_train)}")
    print(f"Testing rows  : {len(X_test)}")
    print(f"\nTest set distribution:")
    print(f"  BUY  (1): {(y_test==1).sum()}")
    print(f"  SELL (0): {(y_test==0).sum()}")

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("\nScaler saved to models/scaler.pkl")

    return X_train_scaled, X_test_scaled


def train_model(X_train, y_train):
    print("\nTraining Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=15,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Cross validation = tests model 5 different ways for reliability
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores : {cv_scores.round(2)}")
    print(f"Average CV accuracy     : {cv_scores.mean()*100:.2f}%")

    print("Training complete!")
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\n{'='*40}")
    print(f"  Model Accuracy : {accuracy * 100:.2f}%")
    print(f"{'='*40}")

    print("\nDetailed Report:")
    print(classification_report(
        y_test, predictions,
        target_names=["SELL", "BUY"],
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["SELL", "BUY"]
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    os.makedirs("reports/Screenshots", exist_ok=True)
    plt.savefig("reports/Screenshots/confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved.")

    return predictions


def save_model(model):
    path = "models/trained_model.pkl"
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def show_feature_importance(model):
    feature_columns = [
        "Open", "High", "Low", "Close", "Volume",
        "MA20", "MA50", "Daily_Return", "Volatility", "Price_Range"
    ]

    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns
    ).sort_values(ascending=False)

    print("\nFeature Importance:")
    print(importance.round(4))

    importance.plot(kind="bar", color="steelblue")
    plt.title("Feature Importance")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("reports/Screenshots/feature_importance.png")
    plt.close()
    print("Feature importance chart saved.")


if __name__ == "__main__":
    os.makedirs("reports/Screenshots", exist_ok=True)

    df                               = load_features()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_s, X_test_s              = scale_features(X_train, X_test)
    model                            = train_model(X_train_s, y_train)
    predictions                      = evaluate_model(model, X_test_s, y_test)
    save_model(model)
    show_feature_importance(model)