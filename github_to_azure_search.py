"""
github_to_azure_search.py

Push model: pulls files from one or more GitHub repos (across one or more
orgs) and pushes them as documents into an Azure AI Search index.

Prerequisites
-------------
1. An Azure AI Search service + an index already created with (at minimum)
   these fields:
       id            (Edm.String, key=true)
       content       (Edm.String, searchable)
       filepath      (Edm.String, filterable)
       repo          (Edm.String, filterable)
       html_url      (Edm.String)
       last_modified (Edm.DateTimeOffset, filterable, sortable)

2. pip install:
       pip install azure-search-documents requests

3. Environment variables:
       GITHUB_TOKEN           -> A token with access to ALL target repos.
                                  The Actions auto-token (secrets.GITHUB_TOKEN)
                                  does NOT work here - it's scoped only to the
                                  repo the pipeline runs in. Use a classic PAT
                                  (repo scope, authorized for SSO on each org)
                                  or a GitHub App installation token instead.
       GH_REPOS           -> Comma-separated list. Each entry is EITHER:
                                    - an explicit "owner/repo", or
                                    - just an org name, e.g. "org2",
                                      which auto-expands to every repo in
                                      that org visible to your token.
                                  Mixing both is fine, e.g.:
                                  "org1/repoA,org2,org3/some-repo"
       GITHUB_BRANCH          -> Branch name applied to every repo above
                                  (optional, defaults to "main"). If different
                                  repos use different default branches, edit
                                  REPO_BRANCH_OVERRIDES below.
       AZURE_SEARCH_ENDPOINT  -> https://<your-service>.search.windows.net
       AZURE_SEARCH_INDEX     -> name of your search index
       AZURE_SEARCH_KEY       -> admin key for your search service

Where to give the GitHub token
-------------------------------
NEVER hardcode the token or repo list in this script. Options, best to worst:

  1. (Best) Azure Key Vault - store GITHUB_TOKEN and AZURE_SEARCH_KEY as
     secrets, pulled at runtime via azure-identity + azure-keyvault-secrets.
  2. GitHub Actions secrets - set GITHUB_TOKEN (your PAT or App token) and
     AZURE_SEARCH_KEY as repo/org secrets, injected as env vars in the
     workflow step. See github-to-azure-search.yml.
  3. Local environment variables, for manual/local runs only.

A single classic PAT can cover multiple orgs, but each org must have
authorized that PAT if it enforces SAML SSO (Org Settings ->
Third-party access -> Personal access tokens). A GitHub App avoids this
by being installed per-org with its own scoped permissions.
"""

import os
import base64
import hashlib
from datetime import datetime, timezone

import requests
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# --------------------------------------------------------------------------
# Config - all pulled from environment variables, nothing hardcoded here.
# --------------------------------------------------------------------------
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]

# Raw entries as given, e.g. "org1/repoA,org2,org3/some-repo" -> a list of
# strings; each is either "owner/repo" or a bare org name to be expanded.
GH_REPOS_RAW = [r.strip() for r in os.environ["GH_REPOS"].split(",") if r.strip()]

DEFAULT_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

# Optional per-repo branch overrides, e.g. {"org2/repoC": "develop"}
REPO_BRANCH_OVERRIDES = {}

AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]
AZURE_SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]

# Only these extensions get indexed. Adjust to your needs.
ALLOWED_EXTENSIONS = {".md", ".py", ".txt", ".json", ".yaml", ".yml", ".ps1", ".tf"}

# Files larger than this are skipped (bytes). Avoids binary/huge files.
MAX_FILE_SIZE_BYTES = 500_000

# Split large text files into chunks of roughly this many characters.
CHUNK_SIZE_CHARS = 4000
CHUNK_OVERLAP_CHARS = 200

GITHUB_API_BASE = "https://api.github.com"


def list_org_repos(org):
    """
    Returns (full_name, default_branch) for every repo the token can see
    in an org, handling pagination. Includes private repos if the token
    has access.
    """
    repos = []
    page = 1
    while True:
        url = f"{GITHUB_API_BASE}/orgs/{org}/repos?per_page=100&page={page}&type=all"
        resp = requests.get(url, headers=github_headers(), timeout=30)
        if resp.status_code == 404:
            print(f"  WARNING: org '{org}' not found or token lacks access - skipping.")
            return []
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        repos.extend(
            (item["full_name"], item.get("default_branch", "main"))
            for item in batch if not item.get("archived")
        )
        page += 1
    return repos


def expand_repo_list(raw_entries):
    """
    Expands a mixed list of "owner/repo" and bare org names into a flat,
    de-duplicated list of (repo, branch) tuples. Explicit "owner/repo"
    entries use DEFAULT_BRANCH / REPO_BRANCH_OVERRIDES; org-expanded
    entries use each repo's actual default branch from the API.
    """
    resolved = []
    for entry in raw_entries:
        if "/" in entry:
            branch = REPO_BRANCH_OVERRIDES.get(entry, DEFAULT_BRANCH)
            resolved.append((entry, branch))
        else:
            print(f"Expanding org '{entry}' to its repos ...")
            org_repos = list_org_repos(entry)
            print(f"  Found {len(org_repos)} repos in '{entry}'")
            resolved.extend(org_repos)

    # de-dupe by repo name while preserving order (first occurrence wins)
    seen = set()
    deduped = []
    for repo, branch in resolved:
        if repo not in seen:
            seen.add(repo)
            deduped.append((repo, branch))
    return deduped


def github_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def list_repo_files(repo, branch):
    """Uses the Git Trees API (recursive=1) to list every file in one repo."""
    url = f"{GITHUB_API_BASE}/repos/{repo}/git/trees/{branch}?recursive=1"
    resp = requests.get(url, headers=github_headers(), timeout=30)
    if resp.status_code == 404:
        print(f"  WARNING: {repo}@{branch} not found or token lacks access - skipping.")
        return []
    resp.raise_for_status()
    tree = resp.json().get("tree", [])

    files = []
    for item in tree:
        if item.get("type") != "blob":
            continue
        path = item["path"]
        _, ext = os.path.splitext(path)
        if ext.lower() not in ALLOWED_EXTENSIONS:
            continue
        if item.get("size", 0) > MAX_FILE_SIZE_BYTES:
            print(f"  Skipping {repo}:{path} (too large: {item.get('size')} bytes)")
            continue
        files.append({"path": path, "sha": item["sha"]})
    return files


def fetch_file_content(repo, sha):
    """Fetches a single blob's content by SHA and decodes it from base64."""
    url = f"{GITHUB_API_BASE}/repos/{repo}/git/blobs/{sha}"
    resp = requests.get(url, headers=github_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("encoding") != "base64":
        return None

    raw_bytes = base64.b64decode(data["content"])
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None


def chunk_text(text, chunk_size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP_CHARS):
    """Simple fixed-size character chunking with overlap."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def make_doc_id(repo, path, chunk_index):
    """
    Azure AI Search document keys allow letters, digits, underscore, dash,
    equal sign. We hash "repo/path" to keep keys short, safe, and unique
    across orgs/repos, then append the chunk index for multi-chunk files.
    """
    digest = hashlib.sha256(f"{repo}/{path}".encode("utf-8")).hexdigest()[:16]
    return f"{digest}_{chunk_index}"


def build_documents_for_repo(repo, branch):
    documents = []
    now = datetime.now(timezone.utc).isoformat()

    files = list_repo_files(repo, branch)
    print(f"  {repo}@{branch}: {len(files)} candidate files")

    for i, f in enumerate(files, 1):
        path = f["path"]
        print(f"    [{i}/{len(files)}] fetching {path} ...")
        content = fetch_file_content(repo, f["sha"])
        if content is None:
            continue

        html_url = f"https://github.com/{repo}/blob/{branch}/{path}"

        for idx, chunk in enumerate(chunk_text(content)):
            documents.append({
                "id": make_doc_id(repo, path, idx),
                "content": chunk,
                "filepath": path,
                "repo": repo,
                "html_url": html_url,
                "last_modified": now,
            })

    return documents


def push_to_azure_search(documents, batch_size=1000):
    if not documents:
        print("No documents to upload.")
        return

    client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        result = client.upload_documents(documents=batch)
        failed = [r for r in result if not r.succeeded]
        print(f"Uploaded batch {i // batch_size + 1}: "
              f"{len(batch) - len(failed)} succeeded, {len(failed)} failed")
        for r in failed:
            print(f"  FAILED key={r.key}: {r.error_message}")


def main():
    repos = expand_repo_list(GH_REPOS_RAW)
    print(f"Processing {len(repos)} repo(s) across their respective orgs:")

    total_docs = 0
    for repo, branch in repos:
        docs = build_documents_for_repo(repo, branch)
        total_docs += len(docs)
        print(f"  Pushing {len(docs)} docs from {repo} to '{AZURE_SEARCH_INDEX}' ...")
        push_to_azure_search(docs)   # push per repo -> incremental progress + no all-or-nothing loss

    print(f"Built and pushed {total_docs} search documents total (after chunking).")
    print("Done.")


if __name__ == "__main__":
    main()
