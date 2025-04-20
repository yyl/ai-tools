# ai-tools

## auto_cat

```
(if start over) sqlite-utils drop-table tools_data/auto_cat/data.db transactions
sqlite-utils insert tools_data/auto_cat/data.db mapped_categories tools_data/auto_cat/category-mapping.json
uv run transaction-processor.py ../data/2024 tools_data/auto_cat/data.db
```