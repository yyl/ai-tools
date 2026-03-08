# ai-tools

## Setup

Requires [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

## auto_cat

```bash
# delete the table if starting over
sqlite-utils drop-table tools_data/auto_cat/data.db transactions

# create the category_mapping table first
sqlite-utils insert tools_data/auto_cat/data.db category_mappings tools_data/auto_cat/category-mapping.json

# process CSV files in the input folder and dump them into transactions table
uv run python transaction-processor.py ../data/2024 tools_data/auto_cat/data.db

# label each transaction using either the existing category if it has one
# against the category_mapping or local llm
# all labels done are also tracked in a history JSON file
uv run python transaction-labeller.py tools_data/auto_cat/data.db
```

## github_repo_stat

Analyzes a public or private GitHub repository and reports code statistics (total lines, files, commits, lifespan, and per-file/per-commit stats).

```bash
# Public repo
uv run python github_repo_stat.py https://github.com/owner/repo

# Private repo (pass token directly)
uv run python github_repo_stat.py https://github.com/owner/private-repo --token ghp_xxx

# Private repo (via environment variable)
GITHUB_TOKEN=ghp_xxx uv run python github_repo_stat.py https://github.com/owner/private-repo
```