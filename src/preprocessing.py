import ast
import pandas as pd
import numpy as np
from src.utils import logger, ensure_dir

NUMERIC_FEATURES = [
    "stargazers_count", "forks_count", "open_issues_count", "watchers_count",
    "size_kb", "topics_count", "age_days", "days_since_push",
    "contributors_count", "commits_count", "releases_count",
    "readme_length", "description_length",
]

BOOL_FEATURES = ["has_wiki", "has_license", "is_template", "has_ci_workflows"]


def _parse_topics(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
    return []


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["topics"] = df["topics"].apply(_parse_topics)
    df["topics_count"] = df["topics"].apply(len)
    df["description"] = df["description"].fillna("").astype(str)
    df["description_length"] = df["description"].str.len()
    df["language"] = df["language"].fillna("Unknown").astype(str)

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in BOOL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    df = df.drop_duplicates(subset=["full_name"])
    df = df[~df["is_fork"]].reset_index(drop=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["log_stars"] = np.log1p(df["stargazers_count"])
    df["log_commits"] = np.log1p(df["commits_count"])
    df["log_contributors"] = np.log1p(df["contributors_count"])
    df["commit_rate"] = df["commits_count"] / df["age_days"].replace(0, 1)
    df["issue_resolution_rate"] = df["forks_count"] / (df["open_issues_count"] + 1)
    df["activity_score"] = (
        df["log_stars"] * 0.3
        + df["log_commits"] * 0.3
        + df["log_contributors"] * 0.2
        + df["releases_count"].clip(0, 50) / 50 * 0.1
        + df["has_ci_workflows"].astype(int) * 0.1
    )

    return df


def preprocess(raw_path: str = "data/raw/repos_raw.csv",
               output_path: str = "data/processed/repos_processed.csv") -> pd.DataFrame:
    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Raw shape: {df.shape}")

    df = clean_raw(df)
    df = engineer_features(df)

    ensure_dir("data/processed")
    df.to_csv(output_path, index=False)
    logger.info(f"Processed {len(df)} repos → {output_path}")
    return df
