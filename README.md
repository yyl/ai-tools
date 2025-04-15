# ai-tools

## auto_cat

```
sqlite-utils ../data/transactions.db "select id, date, amount, description, provider, category from transactions order by id asc limit 10" > ../data/batch1.json
uv run auto_cat.py ../data/batch1.json --model gemma3
sqlite-utils ../data/transactions.db "select id, date, amount, description, provider, category from transactions order by id asc limit 10 offset 10" > ../data/batch2.json
"corrected_category"
```