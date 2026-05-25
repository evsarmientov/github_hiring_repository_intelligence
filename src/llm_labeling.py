import json
import time
import pandas as pd
import anthropic
from tqdm import tqdm
from src.utils import logger, get_env, ensure_dir, CATEGORIES, CATEGORY_DESCRIPTIONS

SYSTEM_PROMPT = """You are an expert engineering manager and technical recruiter evaluating GitHub repositories.
Your task is to classify repositories based on the engineering maturity they reflect — NOT to judge the developer personally.
Be objective and focus on observable signals: code complexity, structure, documentation, community activity, CI/CD, test coverage signals, and project scope."""

LABEL_PROMPT_TEMPLATE = """Classify the following {n} GitHub repositories. For each, assign exactly one category:

CATEGORIES:
- intern: Simple project, typically a first/second programming exercise. Minimal structure, no tests, basic or no README. Single file or very few files.
- junior: Shows some engineering practices but limited experience. Basic tests maybe present, simple architecture, limited documentation. 1-2 years experience level.
- senior: Well-structured, follows best practices, meaningful test coverage signals, CI/CD workflows, proper documentation, clean architecture. 5+ years experience level.
- lead: Complex system design, advanced patterns, multiple contributors, active community, significant releases, framework or library-level scope.
- template: Repository that is clearly a template, boilerplate, starter kit, scaffold, or a near-identical copy/fork of another well-known project. Minimal original work.
- low_value: Essentially empty, abandoned, just config files, a single README with no code, or clearly a toy project with zero engagement.

KEY SIGNALS TO CONSIDER:
- Stars/forks/contributors → community recognition and team size
- Commits/releases → sustained development vs one-off upload
- CI/CD presence → professional engineering practices
- README length → documentation culture
- Repository age vs last push → active vs abandoned
- Topics → project purpose and technical scope
- Description → clarity of purpose

REPOSITORIES TO CLASSIFY:
{repos_block}

Respond ONLY with a valid JSON array with exactly {n} objects, one per repository, in the same order:
[
  {{"id": "owner/repo", "label": "category", "confidence": 0.85, "rationale": "one sentence explanation"}},
  ...
]

Rules:
- "label" must be exactly one of: intern, junior, senior, lead, template, low_value
- "confidence" is a float between 0.0 and 1.0
- "rationale" is a single short sentence
- Do not include any text outside the JSON array"""


def _build_repos_block(rows: list[pd.Series]) -> str:
    parts = []
    for i, row in enumerate(rows, 1):
        parts.append(f"[{i}] {row.get('llm_summary', row.get('bert_input', str(row.get('full_name', ''))))}")
    return "\n\n".join(parts)


def _call_claude(client: anthropic.Anthropic, prompt: str, model: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 10
            logger.warning(f"Rate limit hit. Waiting {wait}s")
            time.sleep(wait)
        except anthropic.APIError as e:
            logger.error(f"API error: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(5)
    raise RuntimeError("Max retries exceeded")


def _parse_llm_response(text: str, expected_ids: list[str]) -> list[dict]:
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array found in response")
    parsed = json.loads(text[start:end])

    results = []
    for item, expected_id in zip(parsed, expected_ids):
        label = item.get("label", "low_value").lower().strip()
        if label not in CATEGORIES:
            label = "low_value"
        results.append({
            "full_name": expected_id,
            "label": label,
            "label_confidence": float(item.get("confidence", 0.5)),
            "label_rationale": item.get("rationale", ""),
        })
    return results


def label_repositories(
    df: pd.DataFrame,
    batch_size: int = 5,
    model: str = "claude-sonnet-4-6",
    output_path: str = "data/labeled/repos_labeled.csv",
) -> pd.DataFrame:
    api_key = get_env("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    all_labels = []
    rows = [row for _, row in df.iterrows()]
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

    logger.info(f"Labeling {len(rows)} repos in {len(batches)} batches (batch_size={batch_size})")

    for batch in tqdm(batches, desc="LLM labeling"):
        repos_block = _build_repos_block(batch)
        ids = [r.get("full_name", f"repo_{i}") for i, r in enumerate(batch)]
        prompt = LABEL_PROMPT_TEMPLATE.format(
            n=len(batch),
            repos_block=repos_block,
        )
        try:
            response_text = _call_claude(client, prompt, model)
            labels = _parse_llm_response(response_text, ids)
            all_labels.extend(labels)
        except Exception as e:
            logger.warning(f"Batch failed ({ids}): {e}. Assigning fallback labels.")
            for repo_id in ids:
                all_labels.append({
                    "full_name": repo_id,
                    "label": "low_value",
                    "label_confidence": 0.0,
                    "label_rationale": "Labeling failed",
                })
        time.sleep(1)

    labels_df = pd.DataFrame(all_labels)
    labeled = df.merge(labels_df[["full_name", "label", "label_confidence", "label_rationale"]], on="full_name", how="left")
    labeled["label"] = labeled["label"].fillna("low_value")
    labeled["label_confidence"] = labeled["label_confidence"].fillna(0.5)

    ensure_dir("data/labeled")
    labeled.to_csv(output_path, index=False)
    logger.info(f"Saved {len(labeled)} labeled repos → {output_path}")
    logger.info(f"Label distribution:\n{labeled['label'].value_counts().to_string()}")
    return labeled


def build_llm_prompt(summary: str, features: dict) -> str:
    return LABEL_PROMPT_TEMPLATE.format(n=1, repos_block=summary)
