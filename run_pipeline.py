"""
Full pipeline orchestrator.

Usage:
    python run_pipeline.py --all
    python run_pipeline.py --stage collect --sample 100
    python run_pipeline.py --stage preprocess
    python run_pipeline.py --stage summarize
    python run_pipeline.py --stage label
    python run_pipeline.py --stage split
    python run_pipeline.py --stage train
    python run_pipeline.py --stage evaluate
    python run_pipeline.py --stage visualize
"""
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))


def stage_collect(args):
    from src.github_collector import collect_repositories
    from src.utils import logger

    logger.info(f"=== STAGE: collect (target={args.sample}) ===")
    collect_repositories(target_count=args.sample)


def stage_preprocess(args):
    from src.preprocessing import preprocess
    from src.utils import logger

    logger.info("=== STAGE: preprocess ===")
    preprocess()


def stage_summarize(args):
    import pandas as pd
    from src.summarization import add_summaries
    from src.utils import logger, ensure_dir

    logger.info("=== STAGE: summarize ===")
    df = pd.read_csv("data/processed/repos_processed.csv")
    df = add_summaries(df)
    ensure_dir("data/processed")
    df.to_csv("data/processed/repos_processed.csv", index=False)
    logger.info(f"Summaries added to {len(df)} repos.")


def stage_label(args):
    import pandas as pd
    from src.llm_labeling import label_repositories
    from src.utils import logger

    logger.info("=== STAGE: label ===")
    df = pd.read_csv("data/processed/repos_processed.csv")
    label_repositories(df, batch_size=5, model=args.model)


def stage_split(args):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from src.utils import logger, ensure_dir, LABEL_MAP

    logger.info("=== STAGE: split ===")
    df = pd.read_csv("data/labeled/repos_labeled.csv")
    df = df.dropna(subset=["label", "bert_input"])
    df = df[df["label"].isin(LABEL_MAP.keys())].reset_index(drop=True)

    logger.info(f"Total labeled samples: {len(df)}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"])

    ensure_dir("data/splits")
    train_df.to_csv("data/splits/train.csv", index=False)
    val_df.to_csv("data/splits/val.csv", index=False)
    test_df.to_csv("data/splits/test.csv", index=False)

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


def stage_train(args):
    from src.train import load_splits, train_model
    from src.utils import logger

    logger.info("=== STAGE: train ===")
    train_df, val_df, test_df = load_splits()
    train_model(train_df, val_df, epochs=args.epochs, batch_size=args.batch_size)


def stage_evaluate(args):
    from src.evaluation import run_full_evaluation
    from src.utils import logger

    logger.info("=== STAGE: evaluate ===")
    run_full_evaluation()


def stage_visualize(args):
    import pandas as pd
    import numpy as np
    from src.visualization import generate_all_figures
    from src.utils import logger, load_json
    from pathlib import Path

    logger.info("=== STAGE: visualize ===")
    df = None
    for p in ["data/labeled/repos_labeled.csv", "data/processed/repos_processed.csv"]:
        if Path(p).exists():
            df = pd.read_csv(p)
            break

    matrix = None
    if Path("output/metrics/confusion_matrix.npy").exists():
        matrix = np.load("output/metrics/confusion_matrix.npy")

    report = None
    if Path("output/metrics/test_metrics.json").exists():
        metrics = load_json("output/metrics/test_metrics.json")
        report = metrics.get("classification_report")

    if df is not None:
        generate_all_figures(df, matrix=matrix, report=report)
        logger.info("Figures saved to output/figures/")
    else:
        logger.error("No data found. Run collect and preprocess first.")


def main():
    parser = argparse.ArgumentParser(description="GitHub Hiring Intelligence Pipeline")
    parser.add_argument("--all", action="store_true", help="Run all stages")
    parser.add_argument("--stage", choices=["collect", "preprocess", "summarize", "label", "split", "train", "evaluate", "visualize"])
    parser.add_argument("--sample", type=int, default=600, help="Number of repos to collect")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Claude model for labeling")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    args = parser.parse_args()

    stages = {
        "collect": stage_collect,
        "preprocess": stage_preprocess,
        "summarize": stage_summarize,
        "label": stage_label,
        "split": stage_split,
        "train": stage_train,
        "evaluate": stage_evaluate,
        "visualize": stage_visualize,
    }

    if args.all:
        for name, fn in stages.items():
            fn(args)
    elif args.stage:
        stages[args.stage](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
