import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from .data_utils import load_dataset
from .modeling import build_models, build_pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "cleaned_scopus_research_abstracts.csv"
ARTIFACT_DIR = ROOT / "artifacts"

def train(data_path=DEFAULT_DATA, test_size=0.2, cv_folds=3, random_state=42):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset(data_path)

    X = data["abstract"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state
    )

    model_definitions = build_models(random_state)
    comparison_rows = []
    fitted_models = {}

    for name, classifier in model_definitions.items():
        print(f"\nCross-validating: {name}")
        pipeline = build_pipeline(classifier)

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
            error_score="raise"
        )

        comparison_rows.append({
            "Model": name,
            "CV Macro-F1 Mean": scores.mean(),
            "CV Macro-F1 Std": scores.std()
        })

        fitted_models[name] = pipeline
        print(f"CV Macro-F1: {scores.mean():.4f} +/- {scores.std():.4f}")

    cv_results = pd.DataFrame(comparison_rows).sort_values(
        "CV Macro-F1 Mean", ascending=False
    ).reset_index(drop=True)

    best_name = cv_results.iloc[0]["Model"]
    best_model = fitted_models[best_name]

    print(f"\nSelected model: {best_name}")
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    labels = sorted(y.unique())
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0))
    }

    report = classification_report(
        y_test, predictions, labels=labels,
        output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()

    cm = confusion_matrix(y_test, predictions, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels]
    )

    cv_results.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)
    report_df.to_csv(ARTIFACT_DIR / "classification_report.csv")
    cm_df.to_csv(ARTIFACT_DIR / "confusion_matrix.csv")

    results = {
        "selected_model": best_name,
        "labels": labels,
        "metrics": metrics,
        "class_counts": data["label"].value_counts().to_dict(),
        "train_class_counts": y_train.value_counts().to_dict(),
        "test_class_counts": y_test.value_counts().to_dict(),
        "dataset_rows": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cv_folds": int(cv_folds),
        "selection_metric": "Mean cross-validation Macro-F1"
    }

    with open(ARTIFACT_DIR / "results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    joblib.dump(best_model, ARTIFACT_DIR / "best_model.joblib")

    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=cv_results,
        x="CV Macro-F1 Mean",
        y="Model",
        color="#4C78A8"
    )
    plt.xlim(0, 1)
    plt.title("Model Comparison: Cross-Validation Macro-F1")
    plt.xlabel("Mean Macro-F1")
    plt.ylabel("Classifier")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "model_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    print("\nTest metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nClassification report:")
    print(report_df)

    print(f"\nArtifacts saved in: {ARTIFACT_DIR}")
    return best_model, cv_results, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train(
        data_path=args.data,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        random_state=args.random_state
    )
