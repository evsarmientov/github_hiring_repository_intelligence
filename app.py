import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Hiring Repository Intelligence",
    layout="wide",
    page_icon="🔍",
)

CATEGORIES = ["intern", "junior", "senior", "lead", "template", "low_value"]
CATEGORY_DESCRIPTIONS = {
    "intern": "Simple project, first/second programming exercise. No tests, basic README.",
    "junior": "Some engineering practices, simple architecture, limited documentation.",
    "senior": "Best practices, CI/CD, good docs, clean architecture.",
    "lead": "Complex design, multiple contributors, active community, significant scope.",
    "template": "Template, boilerplate, starter kit, or near-identical copy.",
    "low_value": "Empty, abandoned, or toy project with zero engagement.",
}
PALETTE = {
    "intern": "#FF6B6B", "junior": "#FFA500", "senior": "#4ECDC4",
    "lead": "#45B7D1", "template": "#96CEB4", "low_value": "#BDBDBD",
}


@st.cache_data
def load_labeled_data():
    for p in ["data/labeled/repos_labeled.csv", "data/processed/repos_processed.csv"]:
        if Path(p).exists():
            return pd.read_csv(p)
    return None


@st.cache_data
def load_metrics():
    p = Path("output/metrics/test_metrics.json")
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_data
def load_confusion_matrix():
    p = Path("output/metrics/confusion_matrix.npy")
    if p.exists():
        return np.load(str(p))
    return None


@st.cache_data
def load_error_analysis():
    p = Path("output/metrics/error_analysis.csv")
    if p.exists():
        return pd.read_csv(p)
    return None


def _fig_to_st(fig):
    st.pyplot(fig)
    plt.close(fig)


def model_loaded() -> bool:
    return (Path("models/trained_models/config.json").exists() or
            Path("models/trained_models/pytorch_model.bin").exists() or
            Path("models/trained_models/model.safetensors").exists())


tabs = st.tabs([
    "Problem & Methodology",
    "Exploratory Analysis",
    "Model Results",
    "Interactive Repository Exploration",
])


with tabs[0]:
    st.header("Problem & Methodology")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Project Objective")
        st.markdown("""
        This system evaluates **GitHub repositories** and classifies them according to the
        **engineering maturity level** they reflect — helping recruiters, engineering managers,
        startups, and technical screening systems make faster, more informed decisions.

        The goal is **NOT** to judge the developer directly. It is to estimate the complexity,
        professionalism, and engineering depth embedded in the repository itself.
        """)

        st.subheader("Repository Selection Methodology")
        st.markdown("""
        Repositories were collected from GitHub's Search API using **stratified sampling** across:
        - **Star count ranges**: 5 000+, 500–5 000, 50–500, 5–50, 0–5 (to capture all maturity levels)
        - **Template/boilerplate keywords**: repos with `template`, `boilerplate`, `starter-kit` in name or description
        - **Low-activity repos**: minimal stars, minimal commits

        **Forks are excluded** to avoid duplicate representations of upstream projects.
        **Selection bias**: highly-starred repos are over-represented in GitHub search results.
        Repos with very few stars may be harder to classify and noisier.
        """)

        st.subheader("GitHub Signals Extracted (13 signals)")
        signals = {
            "stargazers_count": "Community recognition and project popularity",
            "forks_count": "Adoption and derivative usage",
            "contributors_count": "Team size — solo vs collaborative development",
            "commits_count": "Volume of development work",
            "releases_count": "Maturity and versioning practices",
            "has_ci_workflows": "Presence of CI/CD automation (.github/workflows)",
            "readme_length": "Documentation culture and communication",
            "topics_count": "Discoverability and categorization effort",
            "age_days": "Project maturity and longevity",
            "days_since_push": "Activity status — active vs abandoned",
            "open_issues_count": "Community engagement and bug tracking",
            "has_license": "Legal and open-source professionalism",
            "description_length": "Project clarity and communication effort",
        }
        signal_df = pd.DataFrame(
            [(k, v) for k, v in signals.items()],
            columns=["Signal", "Justification"]
        )
        st.dataframe(signal_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Categories")
        for cat, desc in CATEGORY_DESCRIPTIONS.items():
            color = PALETTE[cat]
            st.markdown(
                f'<div style="border-left: 4px solid {color}; padding: 6px 12px; margin-bottom: 8px;">'
                f'<b>{cat.upper()}</b><br><small>{desc}</small></div>',
                unsafe_allow_html=True,
            )

        st.subheader("Dataset Construction")
        df = load_labeled_data()
        if df is not None:
            st.metric("Total repositories", len(df))
            if "label" in df.columns:
                st.metric("Labeled repositories", df["label"].notna().sum())
                st.metric("Unique languages", df["language"].nunique() if "language" in df.columns else "—")
        else:
            st.info("Run the pipeline first to see dataset stats.")

    st.subheader("Prompt Strategy")
    st.markdown("**LLM Prompt Design**: Claude receives a structured summary per repository and is asked to return a JSON array with labels, confidence scores, and rationales.")
    st.code("""System: "You are an expert engineering manager evaluating GitHub repositories..."

User:
Classify the following 5 GitHub repositories...

CATEGORIES:
- intern: Simple project, first/second programming exercise...
- junior: Shows some engineering practices but limited experience...
- senior: Well-structured, follows best practices, CI/CD...
- lead: Complex system design, advanced patterns...
- template: Clearly a template, boilerplate, or starter kit...
- low_value: Empty, abandoned, or toy project...

[1] Repository: torvalds/linux
    Language: C | Stars: 180,000 | Forks: 52,000
    Contributors: 25,000 | Commits: 1,200,000 | Releases: 200
    Has CI/CD: Yes | Has license: Yes | README: 18,000 chars
    ...

Respond ONLY with a valid JSON array:
[{"id": "...", "label": "...", "confidence": 0.9, "rationale": "..."}]""",
        language="text")

    st.subheader("Limitations")
    st.markdown("""
    - **Weak labels**: LLM-generated labels are noisy — the model may misclassify ambiguous repos
    - **Selection bias**: GitHub search favors English-language, high-star repos
    - **Snapshot data**: signals reflect a single point in time, not historical trends
    - **Language gap**: some signals (CI/CD, topics) are less common in non-English ecosystems
    - **Label ambiguity**: boundaries between intern/junior and junior/senior are subjective
    """)


with tabs[1]:
    st.header("Exploratory Analysis")

    df = load_labeled_data()
    if df is None:
        st.warning("No data available. Run the pipeline first: `python run_pipeline.py --all`")
    else:
        from src.visualization import (
            plot_category_distribution, plot_stars_by_category,
            plot_feature_correlation, plot_language_distribution,
            plot_commits_vs_stars,
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Repos", len(df))
        if "language" in df.columns:
            col2.metric("Unique Languages", df["language"].nunique())
        if "stargazers_count" in df.columns:
            col3.metric("Median Stars", int(df["stargazers_count"].median()))
        if "contributors_count" in df.columns:
            col4.metric("Median Contributors", int(df["contributors_count"].median()))

        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Category Distribution")
            if "label" in df.columns:
                fig = plot_category_distribution(df["label"])
                _fig_to_st(fig)
                st.caption(
                    "The distribution shows the relative frequency of each engineering maturity level. "
                    "Imbalanced classes are expected — junior and intern projects are more common on GitHub "
                    "than lead-level open-source systems."
                )
            else:
                st.info("Labels not available yet.")

        with col_b:
            st.subheader("Stars by Category")
            if "label" in df.columns and "stargazers_count" in df.columns:
                fig = plot_stars_by_category(df)
                _fig_to_st(fig)
                st.caption(
                    "Stars correlate with category — senior and lead repos tend to have more community recognition. "
                    "Template repos also attract many stars due to ease of reuse."
                )

        col_c, col_d = st.columns(2)
        with col_c:
            st.subheader("Commits vs Stars")
            if "commits_count" in df.columns and "label" in df.columns:
                fig = plot_commits_vs_stars(df)
                _fig_to_st(fig)
                st.caption(
                    "Lead-level repos cluster in the high-commits, high-stars region. "
                    "Low-value repos concentrate near the origin."
                )

        with col_d:
            st.subheader("Top Programming Languages")
            if "language" in df.columns:
                fig = plot_language_distribution(df)
                _fig_to_st(fig)
                st.caption(
                    "Language distribution reflects the GitHub ecosystem — Python, JavaScript, and TypeScript "
                    "dominate. This may introduce bias toward web and data science projects."
                )

        st.subheader("Feature Correlation Matrix")
        st.caption(
            "Strong positive correlations exist between stars, forks, contributors, and commits — "
            "confirming these signals jointly reflect project maturity. "
            "Days-since-push is negatively correlated with activity signals, as expected."
        )
        if "stargazers_count" in df.columns:
            fig = plot_feature_correlation(df)
            _fig_to_st(fig)

        st.subheader("Signal Statistics by Category")
        if "label" in df.columns:
            numeric_cols = ["stargazers_count", "forks_count", "contributors_count",
                            "commits_count", "releases_count", "readme_length", "has_ci_workflows"]
            available = [c for c in numeric_cols if c in df.columns]
            if available:
                agg = df.groupby("label")[available].median().round(1).reindex(CATEGORIES)
                st.dataframe(agg, use_container_width=True)


with tabs[2]:
    st.header("Model Results")

    metrics = load_metrics()
    matrix = load_confusion_matrix()
    error_df = load_error_analysis()

    if metrics is None:
        st.warning("No model results yet. Run: `python run_pipeline.py --stage train` then `--stage evaluate`")
    else:
        from src.visualization import plot_confusion_matrix, plot_class_f1_comparison

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.3f}")
        col2.metric("Macro F1", f"{metrics.get('test_f1_macro', 0):.3f}")
        col3.metric("Weighted F1", f"{metrics.get('test_f1_weighted', 0):.3f}")
        baseline_f1 = metrics.get("baselines", {}).get("majority_class", {}).get("f1_macro", 0)
        col4.metric("Baseline Macro F1", f"{baseline_f1:.3f}")

        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Confusion Matrix")
            if matrix is not None:
                fig = plot_confusion_matrix(matrix)
                _fig_to_st(fig)
                st.caption("Row = true label, Column = predicted. Diagonal = correct predictions.")

        with col_b:
            st.subheader("Per-Class F1 vs Baseline")
            report = metrics.get("classification_report", {})
            if report:
                fig = plot_class_f1_comparison(report, baseline_f1)
                _fig_to_st(fig)
                st.caption("DistilBERT outperforms the majority-class baseline on most categories.")

        st.subheader("Detailed Classification Report")
        report = metrics.get("classification_report", {})
        if report:
            rows = []
            for cat in CATEGORIES:
                if cat in report:
                    r = report[cat]
                    rows.append({
                        "Category": cat,
                        "Precision": f"{r.get('precision', 0):.3f}",
                        "Recall": f"{r.get('recall', 0):.3f}",
                        "F1-Score": f"{r.get('f1-score', 0):.3f}",
                        "Support": int(r.get("support", 0)),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("Baseline Comparison")
        baselines = metrics.get("baselines", {})
        if baselines:
            b_data = []
            b_data.append({
                "Model": "DistilBERT (fine-tuned)",
                "Accuracy": f"{metrics.get('test_accuracy', 0):.3f}",
                "Macro F1": f"{metrics.get('test_f1_macro', 0):.3f}",
                "Weighted F1": f"{metrics.get('test_f1_weighted', 0):.3f}",
            })
            for strat, vals in baselines.items():
                b_data.append({
                    "Model": f"Baseline ({strat})",
                    "Accuracy": f"{vals.get('accuracy', 0):.3f}",
                    "Macro F1": f"{vals.get('f1_macro', 0):.3f}",
                    "Weighted F1": f"{vals.get('f1_weighted', 0):.3f}",
                })
            st.dataframe(pd.DataFrame(b_data), use_container_width=True, hide_index=True)

        if error_df is not None and len(error_df) > 0:
            st.subheader(f"Error Analysis ({len(error_df)} misclassified examples)")
            st.dataframe(error_df[["full_name", "label", "predicted", "label_rationale"]].head(15),
                         use_container_width=True, hide_index=True)
            st.caption(
                "Common errors occur at the intern/junior and junior/senior boundaries "
                "where signals are ambiguous. Template repos are sometimes confused with low-value repos."
            )

        st.subheader("Discussion")
        gain = metrics.get("comparison", {}).get("bert_vs_majority_f1_gain", 0)
        st.markdown(f"""
        **Key findings**:
        - DistilBERT achieves a **+{gain:.3f} macro F1 improvement** over the majority-class baseline
        - Categories with clear signal profiles (lead, template, low_value) perform best
        - Hardest boundaries: **intern ↔ junior** and **junior ↔ senior** (overlapping signals)

        **Weak points**:
        - LLM-generated labels introduce noise at class boundaries
        - Small class sizes (e.g., lead repos) may underfit
        - Model generalizes from text summaries — doesn't read actual code

        **Possible improvements**:
        - Add code-level signals (file count, test file presence, dependency complexity)
        - Use human-validated labels for a subset to reduce label noise
        - Try longer input sequences with ModernBERT or DeBERTa
        """)


with tabs[3]:
    st.header("Interactive Repository Exploration")

    sub_tabs = st.tabs(["Predict a Repository", "Browse Dataset"])

    with sub_tabs[0]:
        st.subheader("Predict Repository Engineering Maturity")
        st.markdown("Enter a GitHub repository URL or `owner/repo` to analyze it and get a prediction.")

        repo_input = st.text_input("GitHub repository", placeholder="e.g. torvalds/linux or https://github.com/owner/repo")

        if st.button("Analyze", type="primary") and repo_input:
            repo_input = repo_input.strip()
            if "github.com" in repo_input:
                parts = repo_input.rstrip("/").split("github.com/")[-1].split("/")
                if len(parts) >= 2:
                    owner, repo = parts[0], parts[1]
                else:
                    st.error("Could not parse URL. Use format: owner/repo")
                    st.stop()
            elif "/" in repo_input:
                owner, repo = repo_input.split("/", 1)
            else:
                st.error("Please enter in format: owner/repo")
                st.stop()

            with st.spinner(f"Collecting data for {owner}/{repo}..."):
                try:
                    from src.github_collector import get_single_repo
                    from src.preprocessing import clean_raw, engineer_features
                    from src.summarization import build_bert_input, build_llm_summary

                    raw = get_single_repo(owner, repo)
                    if raw is None:
                        st.error("Repository not found or API error.")
                        st.stop()

                    row_df = pd.DataFrame([raw])
                    row_df = clean_raw(row_df)
                    row_df = engineer_features(row_df)
                    row = row_df.iloc[0]

                    bert_text = build_bert_input(row)
                    llm_summary = build_llm_summary(row)

                    col1, col2 = st.columns([3, 2])
                    with col1:
                        st.subheader("Repository Signals")
                        display = {
                            "Stars": int(row.get("stargazers_count", 0)),
                            "Forks": int(row.get("forks_count", 0)),
                            "Contributors": int(row.get("contributors_count", 0)),
                            "Commits": int(row.get("commits_count", 0)),
                            "Releases": int(row.get("releases_count", 0)),
                            "Open issues": int(row.get("open_issues_count", 0)),
                            "Age (days)": int(row.get("age_days", 0)),
                            "Days since push": int(row.get("days_since_push", 0)),
                            "README length": int(row.get("readme_length", 0)),
                            "Has CI/CD": bool(row.get("has_ci_workflows", False)),
                            "Has license": bool(row.get("has_license", False)),
                            "Language": str(row.get("language", "Unknown")),
                            "Topics": row.get("topics_count", 0),
                        }
                        st.dataframe(pd.DataFrame(display.items(), columns=["Signal", "Value"]),
                                     use_container_width=True, hide_index=True)

                    with col2:
                        if model_loaded():
                            from src.train import predict as bert_predict
                            with st.spinner("Running model prediction..."):
                                labels, confidences = bert_predict([bert_text])
                                pred_label = labels[0]
                                confidence = confidences[0]

                            color = PALETTE.get(pred_label, "#888")
                            st.markdown(
                                f'<div style="background: {color}22; border: 2px solid {color}; '
                                f'border-radius: 12px; padding: 20px; text-align: center;">'
                                f'<h2 style="color: {color}; margin: 0;">{pred_label.upper()}</h2>'
                                f'<p style="font-size: 0.9em; margin: 8px 0 0 0;">{CATEGORY_DESCRIPTIONS[pred_label]}</p>'
                                f'<p style="margin: 8px 0 0 0;"><b>Confidence:</b> {confidence:.1%}</p>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.info("Model not trained yet. Showing text summary only.")
                            st.markdown(f"**BERT input text:**\n\n> {bert_text}")

                    with st.expander("View full text summary"):
                        st.text(llm_summary)

                except Exception as e:
                    st.error(f"Error: {e}")

    with sub_tabs[1]:
        st.subheader("Browse Collected Repositories")
        df = load_labeled_data()
        if df is None:
            st.info("No data available yet. Run: `python run_pipeline.py --all`")
        else:
            col_f1, col_f2, col_f3 = st.columns(3)

            with col_f1:
                if "label" in df.columns:
                    cats = ["All"] + CATEGORIES
                    selected_cat = st.selectbox("Category", cats)
                else:
                    selected_cat = "All"

            with col_f2:
                if "language" in df.columns:
                    langs = ["All"] + sorted(df["language"].dropna().unique().tolist())[:30]
                    selected_lang = st.selectbox("Language", langs)
                else:
                    selected_lang = "All"

            with col_f3:
                min_stars = st.number_input("Min stars", min_value=0, value=0, step=10)

            filtered = df.copy()
            if selected_cat != "All" and "label" in filtered.columns:
                filtered = filtered[filtered["label"] == selected_cat]
            if selected_lang != "All" and "language" in filtered.columns:
                filtered = filtered[filtered["language"] == selected_lang]
            if "stargazers_count" in filtered.columns:
                filtered = filtered[filtered["stargazers_count"] >= min_stars]

            st.markdown(f"**{len(filtered)} repositories** matching filters")

            display_cols = [c for c in [
                "full_name", "label", "language", "stargazers_count",
                "contributors_count", "commits_count", "has_ci_workflows",
                "label_rationale"
            ] if c in filtered.columns]

            st.dataframe(
                filtered[display_cols].sort_values("stargazers_count", ascending=False).head(200),
                use_container_width=True,
                hide_index=True,
            )

            if len(filtered) > 0:
                with st.expander("Show example BERT input text"):
                    example = filtered.sample(1).iloc[0]
                    st.markdown(f"**Repository:** `{example.get('full_name', 'unknown')}`")
                    st.markdown(f"**Label:** `{example.get('label', 'unknown')}`")
                    if "bert_input" in example:
                        st.text(example["bert_input"])
                    if "label_rationale" in example:
                        st.markdown(f"**LLM rationale:** {example['label_rationale']}")
