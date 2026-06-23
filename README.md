# TakeMeter

**AI201 · Project 3 — Fine-Tuning a Text Classifier**

TakeMeter classifies r/TrueFilm posts by intent into three categories and compares
a zero-shot LLM baseline against a fine-tuned DistilBERT model.

| Label | Meaning |
|-------|---------|
| `critique` | Long-form, structured analysis of a film's merit or context |
| `observation` | A focused note on a technical element, scene, or performance |
| `inquiry` | A question or prompt meant to start a discussion |

## Repository contents

| File | What it is |
|------|-----------|
| [`planning.md`](planning.md) | Project plan: label taxonomy, data collection, annotation guidelines, modeling & evaluation approach |
| [`ai201_project3_takemeter_starter_clean.ipynb`](ai201_project3_takemeter_starter_clean.ipynb) | Colab notebook: data scraping, fine-tuning pipeline, evaluation, and Groq baseline |
| `data/labeled_dataset.csv` | The 200+ hand-labeled examples (`text`, `label`) used for training/eval |
| `evaluation_results.json` | Accuracy metrics for both models, exported from the notebook |
| `confusion_matrix.png` | Confusion matrix of the fine-tuned model on the test set |

> The CSV, `evaluation_results.json`, and `confusion_matrix.png` are produced by
> running the notebook in Colab and downloading the outputs. See **Reproducing** below.

## Method

- **Data:** up to 250 r/TrueFilm "hot" posts collected via the Reddit API (PRAW),
  filtered to posts with body text, then manually annotated into the three labels.
- **Baseline:** zero-shot classification with Groq `llama-3.3-70b-versatile`
  (`temperature=0`), prompted with the label definitions.
- **Fine-tuned model:** `distilbert-base-uncased` trained for 3 epochs
  (lr `2e-5`, batch size 16) on a 70/15/15 stratified split.
- Both models are evaluated on the same locked test set.

## Results

<!-- Fill in from evaluation_results.json once the Colab run is complete. -->

| Model | Accuracy |
|-------|----------|
| Zero-shot baseline (Groq) | `[baseline_accuracy]` |
| Fine-tuned DistilBERT | `[finetuned_accuracy]` |
| **Improvement** | `[improvement]` |

Test set size: `[test_set_size]` examples.

![Confusion matrix](confusion_matrix.png)

### Evaluation report

<!-- Write your analysis here. Suggested structure: -->
- **Overall:** Did fine-tuning beat the baseline? By how much, and is the gap meaningful given the test set size?
- **Per-class:** Which labels were easiest/hardest? Reference per-class precision/recall.
- **Error analysis:** Pick 3 misclassified test examples (the notebook prints them) and explain *why* each was likely misclassified.
- **Takeaways:** What would you change — more data, clearer label boundaries, a different prompt?

## Reproducing

1. Open the notebook in Colab (badge at the top of the notebook) and select a **T4 GPU** runtime.
2. Add Reddit and Groq credentials as Colab Secrets: `REDDIT_CLIENT_ID`,
   `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`, `GROQ_API_KEY`.
3. Run the scraping cell to produce `truefilm_raw.csv`, annotate it, and upload the
   labeled CSV (or upload `data/labeled_dataset.csv` from this repo).
4. Run the remaining cells to fine-tune, evaluate, and run the baseline.
5. Download `evaluation_results.json` and `confusion_matrix.png` and commit them here.
