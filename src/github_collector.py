import re
import time
import base64
import requests
import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm
from src.utils import logger, get_env, ensure_dir

GITHUB_API = "https://api.github.com"


def _make_headers(token: str = None) -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = token or get_env("GITHUB_TOKEN", required=False)
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get_last_page(link_header: str) -> int:
    if not link_header or 'rel="last"' not in link_header:
        return 1
    m = re.search(r'page=(\d+)>; rel="last"', link_header)
    return int(m.group(1)) if m else 1


def _safe_get(url: str, headers: dict, params: dict = None, timeout: int = 15) -> requests.Response | None:
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        if r.status_code == 403 and "rate limit" in r.text.lower():
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - int(time.time()), 1) + 5
            logger.warning(f"Rate limit hit. Sleeping {wait}s")
            time.sleep(wait)
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r
    except Exception as e:
        logger.debug(f"Request failed for {url}: {e}")
        return None


def _count_via_link(url: str, headers: dict) -> int:
    r = _safe_get(url, headers, params={"per_page": 1})
    if r is None:
        return 0
    last = _get_last_page(r.headers.get("Link", ""))
    if last > 1:
        return last
    try:
        data = r.json()
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def _get_readme_length(owner: str, repo: str, headers: dict) -> int:
    r = _safe_get(f"{GITHUB_API}/repos/{owner}/{repo}/readme", headers)
    if r is None:
        return 0
    try:
        content = base64.b64decode(r.json().get("content", "")).decode("utf-8", errors="ignore")
        return len(content)
    except Exception:
        return 0


def _has_ci_workflows(owner: str, repo: str, headers: dict) -> bool:
    r = _safe_get(f"{GITHUB_API}/repos/{owner}/{repo}/contents/.github/workflows", headers)
    return r is not None and r.status_code == 200


def _enrich_repo(repo: dict, headers: dict) -> dict:
    owner = repo["owner"]["login"]
    name = repo["name"]
    base = f"{GITHUB_API}/repos/{owner}/{name}"

    contributors = _count_via_link(f"{base}/contributors", headers)
    time.sleep(0.3)
    commits = _count_via_link(f"{base}/commits", headers)
    time.sleep(0.3)
    releases = _count_via_link(f"{base}/releases", headers)
    time.sleep(0.3)
    readme_len = _get_readme_length(owner, name, headers)
    time.sleep(0.3)
    has_ci = _has_ci_workflows(owner, name, headers)
    time.sleep(0.3)

    now = datetime.now(timezone.utc)
    created = datetime.fromisoformat(repo.get("created_at", "").replace("Z", "+00:00"))
    pushed = datetime.fromisoformat(repo.get("pushed_at", "").replace("Z", "+00:00"))

    topics = repo.get("topics", [])

    return {
        "full_name": repo["full_name"],
        "owner": owner,
        "repo_name": name,
        "description": repo.get("description") or "",
        "language": repo.get("language") or "Unknown",
        "stargazers_count": repo.get("stargazers_count", 0),
        "forks_count": repo.get("forks_count", 0),
        "open_issues_count": repo.get("open_issues_count", 0),
        "watchers_count": repo.get("watchers_count", 0),
        "size_kb": repo.get("size", 0),
        "topics": topics,
        "topics_count": len(topics),
        "has_wiki": repo.get("has_wiki", False),
        "has_license": repo.get("license") is not None,
        "is_template": repo.get("is_template", False),
        "is_fork": repo.get("fork", False),
        "default_branch": repo.get("default_branch", "main"),
        "created_at": repo.get("created_at", ""),
        "pushed_at": repo.get("pushed_at", ""),
        "age_days": (now - created).days,
        "days_since_push": (now - pushed).days,
        "contributors_count": contributors,
        "commits_count": commits,
        "releases_count": releases,
        "readme_length": readme_len,
        "has_ci_workflows": has_ci,
        "description_length": len(repo.get("description") or ""),
    }


SEARCH_QUERIES = [
    ("stars:5000..50000 fork:false", 80),
    ("stars:500..5000 fork:false", 80),
    ("stars:50..500 fork:false", 80),
    ("stars:5..50 fork:false", 80),
    ("stars:0..5 fork:false", 60),
    ('template OR boilerplate OR starter-kit fork:false stars:>10', 60),
    ('in:name template fork:false stars:0..50', 40),
    ('in:name boilerplate fork:false stars:0..20', 40),
]


def search_repositories(query: str, max_results: int, headers: dict) -> list[dict]:
    repos = []
    per_page = min(100, max_results)
    page = 1
    seen = set()

    while len(repos) < max_results:
        r = _safe_get(
            f"{GITHUB_API}/search/repositories",
            headers,
            params={"q": query, "sort": "updated", "order": "desc", "per_page": per_page, "page": page},
        )
        if r is None:
            break
        items = r.json().get("items", [])
        if not items:
            break
        for item in items:
            if item["full_name"] not in seen:
                seen.add(item["full_name"])
                repos.append(item)
        if len(items) < per_page:
            break
        page += 1
        time.sleep(2)

    return repos[:max_results]


def collect_repositories(target_count: int = 600, token: str = None) -> pd.DataFrame:
    headers = _make_headers(token)
    all_raw = []
    seen = set()

    for query, quota in SEARCH_QUERIES:
        needed = min(quota, target_count - len(seen))
        if needed <= 0:
            break
        logger.info(f"Searching: '{query}' (target {needed})")
        results = search_repositories(query, needed, headers)
        for r in results:
            if r["full_name"] not in seen:
                seen.add(r["full_name"])
                all_raw.append(r)
        logger.info(f"  → {len(results)} found. Total unique: {len(seen)}")
        time.sleep(2)

    logger.info(f"Enriching {len(all_raw)} repositories with detailed signals...")
    enriched = []
    for repo in tqdm(all_raw, desc="Enriching repos"):
        try:
            data = _enrich_repo(repo, headers)
            enriched.append(data)
        except Exception as e:
            logger.warning(f"Skipping {repo.get('full_name', '?')}: {e}")
        time.sleep(0.5)

    df = pd.DataFrame(enriched)
    ensure_dir("data/raw")
    df.to_csv("data/raw/repos_raw.csv", index=False)
    logger.info(f"Saved {len(df)} repos to data/raw/repos_raw.csv")
    return df


def get_single_repo(owner: str, repo: str, token: str = None) -> dict | None:
    headers = _make_headers(token)
    r = _safe_get(f"{GITHUB_API}/repos/{owner}/{repo}", headers)
    if r is None:
        return None
    return _enrich_repo(r.json(), headers)
