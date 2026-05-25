import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.utils import ensure_dir, CATEGORIES

PALETTE = {
    "intern": "#FF6B6B",
    "junior": "#FFA500",
    "senior": "#4ECDC4",
    "lead": "#45B7D1",
    "template": "#96CEB4",
    "low_value": "#BDBDBD",
}
CATEGORY_ORDER = CATEGORIES


def _save(fig, path: str):
    ensure_dir(str(Path(path).parent))
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_category_distribution(labels, title: str = "Label Distribution") -> plt.Figure:
    counts = pd.Series(labels).value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE.get(c, "#888") for c in counts.index])
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.15)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(matrix: np.ndarray, labels: list = CATEGORIES) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    matrix_norm = matrix.astype(float) / matrix.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(
        matrix_norm,
        annot=matrix,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Confusion Matrix (normalized by true label)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_stars_by_category(df: pd.DataFrame) -> plt.Figure:
    plot_df = df[df["label"].isin(CATEGORY_ORDER)].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=plot_df,
        x="label",
        y="stargazers_count",
        order=CATEGORY_ORDER,
        palette=PALETTE,
        ax=ax,
        showfliers=False,
    )
    ax.set_yscale("symlog")
    ax.set_title("Stars by Category (log scale, outliers hidden)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Stars")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig


def plot_feature_correlation(df: pd.DataFrame) -> plt.Figure:
    numeric_cols = [
        "stargazers_count", "forks_count", "contributors_count",
        "commits_count", "releases_count", "readme_length",
        "age_days", "days_since_push", "topics_count", "has_ci_workflows",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    corr = df[available].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                linewidths=0.3, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    return fig


def plot_language_distribution(df: pd.DataFrame, top_n: int = 12) -> plt.Figure:
    top_langs = df["language"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top_langs.index[::-1], top_langs.values[::-1], color="#45B7D1")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title(f"Top {top_n} Programming Languages", fontsize=13, fontweight="bold")
    ax.set_xlabel("Count")
    plt.tight_layout()
    return fig


def plot_class_f1_comparison(report: dict, baseline_f1: float) -> plt.Figure:
    categories = [c for c in CATEGORY_ORDER if c in report]
    bert_f1 = [report[c]["f1-score"] for c in categories]
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, bert_f1, width, label="DistilBERT", color="#45B7D1")
    ax.axhline(baseline_f1, color="#FF6B6B", linestyle="--", linewidth=1.5, label=f"Majority Baseline F1={baseline_f1:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class F1 Score vs Baseline", fontsize=13, fontweight="bold")
    ax.set_ylabel("F1 Score")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_commits_vs_stars(df: pd.DataFrame) -> plt.Figure:
    plot_df = df[df["label"].isin(CATEGORY_ORDER)].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    for cat in CATEGORY_ORDER:
        sub = plot_df[plot_df["label"] == cat]
        ax.scatter(
            sub["commits_count"].clip(upper=5000),
            sub["stargazers_count"].clip(upper=50000),
            label=cat, color=PALETTE.get(cat, "#888"),
            alpha=0.6, s=20,
        )
    ax.set_xlabel("Commits (clipped at 5k)")
    ax.set_ylabel("Stars (clipped at 50k)")
    ax.set_title("Commits vs Stars by Category", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig


def generate_all_figures(df: pd.DataFrame, matrix: np.ndarray = None, report: dict = None):
    ensure_dir("output/figures")

    fig = plot_category_distribution(df["label"] if "label" in df.columns else [])
    _save(fig, "output/figures/category_distribution.png")

    fig = plot_feature_correlation(df)
    _save(fig, "output/figures/feature_correlation.png")

    if "stargazers_count" in df.columns and "label" in df.columns:
        fig = plot_stars_by_category(df)
        _save(fig, "output/figures/stars_by_category.png")

    if "language" in df.columns:
        fig = plot_language_distribution(df)
        _save(fig, "output/figures/language_distribution.png")

    if "commits_count" in df.columns and "label" in df.columns:
        fig = plot_commits_vs_stars(df)
        _save(fig, "output/figures/commits_vs_stars.png")

    if matrix is not None:
        fig = plot_confusion_matrix(matrix)
        _save(fig, "output/figures/confusion_matrix.png")

    if report is not None:
        baseline_f1 = 0.0
        fig = plot_class_f1_comparison(report, baseline_f1)
        _save(fig, "output/figures/class_f1_comparison.png")
