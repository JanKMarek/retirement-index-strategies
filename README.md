# Retirement Index Strategies

This project explores index-based retirement index strategies.

## Workflows

### Install
```
uv sync
```

### Keep local environment and Databricks in synch 
- uv add <package>
- uv export --format requirements.txt > requirements.txt
- git add, commit, push to GitHub
- Databricks: pull changes in the Git Folder
- %pip install -r requirements.txt

### Usage
```bash
uv run retirement-index-strategies
```

### Testing
```bash
uv run pytest
```
