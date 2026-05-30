import ast
import pandas as pd


def _parse_topics(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return [t.strip() for t in val.split(",") if t.strip()]
    return []


def build_llm_summary(row: pd.Series) -> str:
    topics = _parse_topics(row.get("topics", []))
    topics_str = ", ".join(topics[:8]) if topics else "none"
    has_ci = "Yes" if row.get("has_ci_workflows", False) else "No"
    has_lic = "Yes" if row.get("has_license", False) else "No"
    has_wiki = "Yes" if row.get("has_wiki", False) else "No"
    desc = str(row.get("description", "") or "")

    return (
        f"Repository: {row.get('full_name', 'unknown')}\n"
        f"Description: {desc[:200] or 'None'}\n"
        f"Language: {row.get('language', 'Unknown')}\n"
        f"Stars: {int(row.get('stargazers_count', 0))} | "
        f"Forks: {int(row.get('forks_count', 0))} | "
        f"Open issues: {int(row.get('open_issues_count', 0))}\n"
        f"Contributors: {int(row.get('contributors_count', 0))} | "
        f"Commits: {int(row.get('commits_count', 0))} | "
        f"Releases: {int(row.get('releases_count', 0))}\n"
        f"Age: {int(row.get('age_days', 0))} days | "
        f"Last push: {int(row.get('days_since_push', 0))} days ago\n"
        f"Topics: {topics_str}\n"
        f"Has CI/CD: {has_ci} | Has license: {has_lic} | Has wiki: {has_wiki}\n"
        f"README length: {int(row.get('readme_length', 0))} chars | "
        f"Description length: {int(row.get('description_length', 0))} chars"
    )


def build_bert_input(row: pd.Series) -> str:
    topics = _parse_topics(row.get("topics", []))
    topics_str = " ".join(topics[:6]) if topics else "no topics"
    has_ci = "has CI/CD workflows" if row.get("has_ci_workflows", False) else "no CI/CD"
    has_lic = "has a license" if row.get("has_license", False) else "no license"
    desc = str(row.get("description", "") or "")
    desc_part = f'Description: "{desc[:150]}". ' if desc else ""

    stars = int(row.get("stargazers_count", 0))
    forks = int(row.get("forks_count", 0))
    issues = int(row.get("open_issues_count", 0))
    contributors = int(row.get("contributors_count", 0))
    commits = int(row.get("commits_count", 0))
    releases = int(row.get("releases_count", 0))
    age = int(row.get("age_days", 0))
    last_push = int(row.get("days_since_push", 0))
    readme = int(row.get("readme_length", 0))
    lang = row.get("language", "Unknown")

    return (
        f"{desc_part}"
        f"This {lang} repository has {stars} stars, {forks} forks, and {issues} open issues. "
        f"It has {contributors} contributors and {commits} commits with {releases} releases. "
        f"The project is {age} days old with last activity {last_push} days ago. "
        f"Topics: {topics_str}. "
        f"It {has_ci} and {has_lic}. "
        f"README is {readme} characters long."
    )


def add_summaries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["llm_summary"] = df.apply(build_llm_summary, axis=1)
    df["bert_input"] = df.apply(build_bert_input, axis=1)
    return df


def summarize_repository(metadata: dict) -> str:
    row = pd.Series(metadata)
    return build_bert_input(row)
