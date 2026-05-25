import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from src.utils import logger, ensure_dir, save_json, LABEL_MAP, CATEGORIES


def evaluate_predictions(y_true: list, y_pred: list) -> tuple[dict, np.ndarray]:
    report = classification_report(y_true, y_pred, labels=CATEGORIES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=CATEGORIES)
    return report, matrix


def baseline_majority(y_train: list, y_test: list) -> dict:
    majority = pd.Series(y_train).value_counts().idxmax()
    y_pred = [majority] * len(y_test)
    return {
        "strategy": "majority_class",
        "majority_label": majority,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", labels=CATEGORIES, zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", labels=CATEGORIES, zero_division=0),
    }


def baseline_random(y_train: list, y_test: list) -> dict:
    rng = np.random.default_rng(42)
    dist = pd.Series(y_train).value_counts(normalize=True)
    classes = dist.index.tolist()
    probs = dist.values
    y_pred = rng.choice(classes, size=len(y_test), p=probs).tolist()
    return {
        "strategy": "stratified_random",
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", labels=CATEGORIES, zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", labels=CATEGORIES, zero_division=0),
    }


def error_analysis(
    df_test: pd.DataFrame, y_pred: list, n_errors: int = 20
) -> pd.DataFrame:
    df = df_test.copy()
    df["predicted"] = y_pred
    df["correct"] = df["label"] == df["predicted"]
    wrong = df[~df["correct"]].copy()
    cols = ["full_name", "label", "predicted", "label_rationale", "bert_input"]
    available = [c for c in cols if c in wrong.columns]
    return wrong[available].head(n_errors).reset_index(drop=True)


def run_full_evaluation(
    model_dir: str = "models/trained_models",
    splits_dir: str = "data/splits",
    output_dir: str = "output/metrics",
):
    from src.train import load_splits, encode_labels, predict

    ensure_dir(output_dir)
    train_df, val_df, test_df = load_splits(splits_dir)

    logger.info("Generating predictions on test set...")
    test_texts = test_df["bert_input"].astype(str).tolist()
    y_pred, confidences = predict(test_texts, model_dir)
    y_true = test_df["label"].tolist()

    report, matrix = evaluate_predictions(y_true, y_pred)
    test_df["predicted"] = y_pred
    test_df["confidence"] = confidences

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=CATEGORIES, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=CATEGORIES, zero_division=0)

    logger.info(f"Test Accuracy: {accuracy:.3f} | F1 Macro: {f1_macro:.3f} | F1 Weighted: {f1_weighted:.3f}")

    train_labels = train_df["label"].tolist()
    baseline_maj = baseline_majority(train_labels, y_true)
    baseline_rnd = baseline_random(train_labels, y_true)

    summary = {
        "test_accuracy": accuracy,
        "test_f1_macro": f1_macro,
        "test_f1_weighted": f1_weighted,
        "classification_report": report,
        "baselines": {
            "majority_class": baseline_maj,
            "stratified_random": baseline_rnd,
        },
        "comparison": {
            "bert_vs_majority_accuracy_gain": accuracy - baseline_maj["accuracy"],
            "bert_vs_majority_f1_gain": f1_macro - baseline_maj["f1_macro"],
        },
    }

    save_json(summary, f"{output_dir}/test_metrics.json")
    np.save(f"{output_dir}/confusion_matrix.npy", matrix)

    error_df = error_analysis(test_df, y_pred)
    error_df.to_csv(f"{output_dir}/error_analysis.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_predictions.csv", index=False)

    logger.info(f"Saved evaluation results to {output_dir}")
    return summary, matrix, error_df
