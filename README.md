# ai-tools

## auto_cat

```
# delete the table if already exists
(if start over) sqlite-utils drop-table tools_data/auto_cat/data.db transactions

# create the category_mapping table first
sqlite-utils insert tools_data/auto_cat/data.db category_mappings tools_data/auto_cat/category-mapping.json

# process CSV files in the input folder and dump them into transactions table
uv run transaction-processor.py ../data/2024 tools_data/auto_cat/data.db

# label each transaction using either the existing category if it has one against the category_mapping or local llm
# it writes to update the existing transaction in the database
# all labels done are also tracked in a history JSON file
uv run transaction-labeller.py tools_data/auto_cat/data.db
```