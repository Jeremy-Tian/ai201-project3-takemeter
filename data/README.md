# data/

## Workflow

1. **Collect** real public posts into a scaffold:
   ```
   python3 collect_posts.py --target 300 --out data/truefilm_raw.csv
   ```
   (Or use the notebook's PRAW scraping cell in Colab if public JSON is blocked
   on your network.) This produces `truefilm_raw.csv` with empty `label`/`notes`.

2. **Annotate** by reading each post and filling the `label` column using the
   definitions in [`planning.md`](../planning.md) §2. Log genuinely ambiguous
   cases in `notes`. Aim for **200+ examples**, no label over 70% of the total
   (see planning.md §4).

3. **Save** the finished file as `labeled_dataset.csv` here and commit it. This is
   the file the notebook expects — one complete labeled CSV, not pre-split.

## `labeled_dataset.csv` columns

| Column | Required | Meaning |
|--------|----------|---------|
| `text` | yes | The post body. |
| `label` | yes | One of `critique`, `observation`, `inquiry`. |
| `notes` | yes (this project) | Notes on difficult/ambiguous cases. |
| `prelabeled_by_ai` | optional | `True` if an LLM pre-labeled this row (see planning.md §7.2). |
| `human_corrected` | optional | `True` if your review changed the pre-label. |

The notebook only reads `text` and `label`; the extra columns are for your
annotation tracking and AI-usage disclosure.
