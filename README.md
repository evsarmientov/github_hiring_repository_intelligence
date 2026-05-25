# GitHub Hiring Repository Intelligence

**Track A — Hiring-Oriented Repository Intelligence**

A weak-supervision NLP pipeline that classifies GitHub repositories by engineering maturity level, designed to help recruiters, engineering managers, and technical screening systems evaluate repositories quickly and objectively.

---

## What does this project do?

This system analyzes GitHub repositories and classifies them into one of six engineering maturity categories:

| Category | Description |
|---|---|
| `intern` | Simple project, first/second programming exercise, minimal structure |
| `junior` | Some engineering practices, simple architecture, limited documentation |
| `senior` | Best practices, CI/CD, good documentation, clean architecture |
| `lead` | Complex design, active community, multiple contributors, advanced scope |
| `template` | Boilerplate, starter kit, or near-identical copy of another project |
| `low_value` | Empty, abandoned, or toy project with no meaningful engagement |

---

## Pipeline Architecture

```
GitHub API → Preprocessing → Text Summarization → LLM Weak Labeling → Train/Val/Test Split → DistilBERT Fine-tuning → Evaluation
```

### Stage 1 — GitHub Data Collection
Repositories are collected via GitHub Search API with **stratified sampling** across star-count ranges (0–5, 5–50, 50–500, 500–5000, 5000+) and template/boilerplate keyword searches. Forks are excluded. Minimum 13 signals are extracted per repository.

**Repository signals collected:**
- `stargazers_count`, `forks_count`, `open_issues_count`, `watchers_count`
- `contributors_count` (via Link header pagination)
- `commits_count` (via Link header pagination)
- `releases_count`
- `has_ci_workflows` (presence of `.github/workflows/`)
- `readme_length` (decoded README character count)
- `topics_count`, `description_length`, `has_license`, `age_days`, `days_since_push`

### Stage 2 — Repository Representation
Each repository is converted into a natural-language summary for BERT input:
```
Description: "Fast async HTTP client". This Python repository has 4523 stars, 312 forks,
and 18 open issues. It has 8 contributors and 1240 commits with 23 releases.
The project is 1200 days old with last activity 3 days ago. Topics: async, http, python.
It has CI/CD workflows and a license. README is 8420 characters long.
```

### Stage 3 — Weak Labeling with Claude
Claude (claude-sonnet-4-6) receives batches of 5 repositories with their structured summaries and returns JSON labels with confidence scores and rationales. Prompt is carefully designed with category definitions and key signals to consider.

### Stage 4 — Train / Validation / Test Split
- **70%** train, **15%** validation, **15%** test
- Stratified split to preserve class distribution

### Stage 5 — BERT Fine-tuning
**Model:** `distilbert-base-uncased` (66M parameters, lightweight)
- Input: text summary (max 256 tokens)
- Output: 6-class classification
- Class-weighted cross-entropy loss (handles class imbalance)
- Early stopping on validation macro F1

### Stage 6 — Evaluation & Error Analysis
Metrics: Accuracy, Precision, Recall, F1 (per-class and macro/weighted)
Baseline comparison: majority-class and stratified-random baselines
Error analysis: common misclassification patterns at class boundaries

---

## Final Metrics

See `output/metrics/test_metrics.json` after running the pipeline.

---

## Main Limitations

- **Label noise**: LLM-generated labels are noisy at class boundaries (intern↔junior, junior↔senior)
- **Selection bias**: GitHub search favors English-language repos and popular technologies
- **Snapshot signals**: metrics reflect one point in time, not longitudinal trends
- **No code analysis**: model reads text summaries, not actual source code

---

## Business Applications

- **Recruiting automation**: pre-screen candidate portfolios before technical interviews
- **Startup hiring**: small teams without dedicated HR can assess engineer level quickly
- **Accelerator intake**: evaluate founding team technical depth from public repos
- **Engineering manager tooling**: benchmark candidates against engineering maturity expectations

---

## How to Run

### 1. Setup environment
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your GITHUB_TOKEN and ANTHROPIC_API_KEY
```

### 2. Run the full pipeline
```bash
# Full pipeline (collects 600 repos)
python run_pipeline.py --all

# Or run individual stages:
python run_pipeline.py --stage collect --sample 600
python run_pipeline.py --stage preprocess
python run_pipeline.py --stage summarize
python run_pipeline.py --stage label
python run_pipeline.py --stage split
python run_pipeline.py --stage train
python run_pipeline.py --stage evaluate
python run_pipeline.py --stage visualize
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

---

## Video

See `video/link.txt` for the presentation video.
