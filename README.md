# TakeMeter

**AI201 · Project 3 — Fine-Tuning a Text Classifier**

TakeMeter classifies r/TrueFilm posts by *intent* into three categories and
compares a zero-shot LLM baseline against a fine-tuned DistilBERT model.

> **Status of this README:** sections that describe design decisions and method
> are complete. Sections marked **⬜ FILL IN** require your actual Colab run —
> run the notebook, download `evaluation_results.json` and `confusion_matrix.png`,
> then `python3 make_report.py` to generate the results/confusion-matrix/per-class
> tables and paste them in. Do **not** ship this with the fill-in blocks remaining.

---

## 1. Project overview

Posts in [r/TrueFilm](https://www.reddit.com/r/TrueFilm/) vary widely in intent:
some are long-form critical essays, some are narrow technical notes, and some exist
mainly to start a discussion. TakeMeter assigns each post one of three intent labels
so the community could filter or route posts by type. Full rationale and the
pre-annotation design are in [`planning.md`](planning.md).

| Label | Meaning |
|-------|---------|
| `critique` | A long-form, structured argument analyzing a film's narrative, themes, context, or overall merit as a whole work. |
| `observation` | A focused comment on a single technical element, one scene, or one performance, rather than the whole work. |
| `inquiry` | A post whose primary purpose is to ask a question or prompt community discussion. |

**Two examples per label:**

- **`critique`**
  - *"Across his late period, Kim Ki-duk uses color not as decoration but as a moral register — the shift to muted palettes tracks each protagonist's loss of innocence, and reading the films chronologically makes that arc unmistakable."*
  - *"Tarkovsky's long takes aren't slow for their own sake; they force the viewer into the film's own sense of time, which is why summarizing the plot of Stalker tells you nothing about what the film actually is."*
- **`observation`**
  - *"I only noticed on rewatch that the blocking in the dinner scene mirrors a stage play — everyone stays on their side of the table until the accusation lands."*
  - *"The match cut from the bone to the spacecraft is famous, but the sound edit underneath it is what actually sells the jump."*
- **`inquiry`**
  - *"Why do you think the French New Wave still resonates with modern independent filmmakers, when so many other movements feel dated?"*
  - *"What's a film you initially disliked but came to admire after a second viewing, and what changed for you?"*

## 2. Repository contents

| File | What it is |
|------|-----------|
| [`planning.md`](planning.md) | Pre-annotation plan: community, labels, edge cases, data plan, metrics, success criteria, AI tool plan |
| [`ai201_project3_takemeter_starter_clean.ipynb`](ai201_project3_takemeter_starter_clean.ipynb) | Colab notebook: scraping, fine-tuning, evaluation, Groq baseline |
| [`collect_posts.py`](collect_posts.py) | Local collector → `data/truefilm_raw.csv` scaffold for manual annotation |
| `data/labeled_dataset.csv` | The hand-labeled examples (`text`, `label`, `notes`) |
| [`make_report.py`](make_report.py) | Renders `evaluation_results.json` into README markdown tables |
| `evaluation_results.json` | Metrics + confusion matrices, exported from the notebook |
| `confusion_matrix.png` | Confusion matrix image (supplementary to the markdown table below) |

## 3. Dataset

- **Source:** public r/TrueFilm posts with body text, collected via the Reddit
  public JSON / API across the hot, top, new, and rising sortings (the multi-sorting
  spread is to surface `inquiry`-type posts that rarely reach "hot"). See
  [`collect_posts.py`](collect_posts.py).
- **Labeling process:** each post was read individually and assigned the single
  label matching its *primary intent* using the §1 definitions and the
  `critique`/`observation` scope rule and `inquiry` "remove-the-question" test from
  `planning.md` §3. Genuinely ambiguous cases were recorded in a `notes` column so
  the same tie-break was applied consistently whenever the pattern recurred. The
  test set was labeled by hand without any AI pre-labeling (see §7).

> **⬜ FILL IN — dataset stats** (count from your `labeled_dataset.csv`):
> - Total labeled examples: `[N]`
> - Distribution: critique `[n]` · observation `[n]` · inquiry `[n]`
> - Largest class share: `[xx%]` (must be ≤ 70% — see planning.md §4)

**Three difficult-to-label examples and my decisions — ⬜ FILL IN** (pull real ones
from your `notes` column):

| # | Post (excerpt) | Candidate labels | Decision + reasoning |
|---|----------------|------------------|----------------------|
| 1 | `[paste]` | `[e.g. critique vs observation]` | `[which I chose and why, per the scope rule]` |
| 2 | `[paste]` | | |
| 3 | `[paste]` | | |

## 4. Method

### 4.1 Fine-tuned model

- **Base model:** `distilbert-base-uncased` with a 3-class sequence-classification head.
- **Training setup:** 70/15/15 stratified train/val/test split (`random_state=42`),
  tokenized at `max_length=256`, trained for 3 epochs with the best checkpoint
  selected by validation accuracy. Evaluation is on the **locked test set** that
  neither model sees during training/prompting.
- **Hyperparameters:** 3 epochs, learning rate `2e-5`, train batch size 16, weight
  decay `0.01`, 50 warmup steps.

**Hyperparameter decision (≥1 required):** I kept the learning rate at **`2e-5`**
and epochs at **3** rather than training longer. `2e-5` is the standard fine-tuning
rate for BERT-family models, and with only ~200 examples, more epochs mainly risk
**overfitting** to a small training set — exactly the failure the §5.4 reflection
checks for. I treated 3 epochs / `2e-5` as a deliberate floor and would only have
increased them if validation accuracy were still climbing at epoch 3.

> **⬜ FILL IN:** if you changed any default after seeing your run, state what and
> why; otherwise confirm you used the justified defaults above.

### 4.2 Baseline (zero-shot)

- **Model:** Groq `llama-3.3-70b-versatile`, `temperature=0`, `max_tokens=20`.
- **How results were collected:** every post in the **same locked test set** is sent
  to the model one at a time with the system prompt below; the response is lowercased
  and matched against the valid label names (longest-first to avoid substring
  collisions). Responses that match no label are counted as **unparseable and
  excluded** from accuracy. (If >~10% are unparseable, the prompt needs tightening.)
- **Prompt used** (Section 5 of the notebook, aligned to `planning.md` §2–3):

```
You are classifying posts from the r/TrueFilm community by their PRIMARY intent.
Assign each post to exactly one of the following three categories.

critique: A long-form, structured argument that analyzes a film's narrative, themes,
historical context, or overall artistic merit as a whole work.
observation: A focused comment on a single technical element, one specific scene, or
an individual performance, rather than the work as a whole.
inquiry: A post whose primary purpose is to ask a question or prompt community
discussion about a director, genre, film, or idea.

How to decide between close cases:
- If the analysis is anchored to a single scene, shot, performance, or technical
  choice, label it observation. If it makes a claim about the work as a whole,
  label it critique.
- If removing the question still leaves a complete, self-standing argument, it is
  critique. If removing the question leaves nothing substantive, it is inquiry.

Respond with ONLY the label name: critique, observation, or inquiry.
```

(Each label in the prompt also carries one worked example; see the notebook cell for
the full text.)

---

## 5. Evaluation report

### 5.1 Results

> **⬜ FILL IN** — run `python3 make_report.py` and paste its output here. It
> produces the results table, per-class metrics for **both** models, and the
> confusion matrices as markdown tables. Template below:

| Model | Accuracy |
|-------|----------|
| Zero-shot baseline (Groq) | `[baseline_accuracy]` |
| Fine-tuned DistilBERT | `[finetuned_accuracy]` |
| **Improvement** | `[improvement]` |

Test set size: `[test_set_size]` examples.

**Per-class metrics (fine-tuned) — ⬜ FILL IN from make_report.py**

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| critique | | | | |
| observation | | | | |
| inquiry | | | | |

**Per-class metrics (baseline) — ⬜ FILL IN from make_report.py**

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| critique | | | | |
| observation | | | | |
| inquiry | | | | |

**Confusion matrix — fine-tuned model (⬜ FILL IN, rows = true, cols = predicted):**

| true \ pred | critique | observation | inquiry |
|---|---|---|---|
| **critique** | | | |
| **observation** | | | |
| **inquiry** | | | |

`confusion_matrix.png` is committed as a supplementary copy:

![Confusion matrix](confusion_matrix.png)

### 5.2 Error analysis

> **AI-assisted pattern surfacing (planning.md §7.3):** paste the notebook's list of
> wrong predictions into an LLM and ask it to find common themes — confused label
> pair, post length, sarcasm, short/low-information posts. Then **re-read the actual
> examples to verify** each pattern before writing it up.
>
> **⬜ FILL IN:** What patterns did the AI suggest? Which did you confirm by
> re-reading, and which did you **discard** as not holding up? (Disclose both.)

> **⬜ FILL IN — at least 3 specific misclassified examples.** For each, go beyond
> "the model got it wrong" using these questions:
> - **Which labels are confused?** Read the confusion matrix — is one directional
>   pair (e.g. critique → observation) most of the errors? That names the boundary
>   the model hasn't learned.
> - **Why is that boundary hard?** Ambiguous language, sarcasm, short posts, or
>   topic-vs-structure mismatch?
> - **Labeling problem or data/prompt problem?** If you labeled consistently and it
>   still fails → training-data distribution or the boundary itself. If you find you
>   labeled similar posts differently → annotation inconsistency.
> - **What would fix it?** More examples for the confused class, a tighter label
>   definition, or more diverse examples of the hard case?

**Example 1** — true: `[ ]`, predicted: `[ ]` (confidence `[ ]`)
> Post: `[paste]`
> Analysis: `[...]`

**Example 2** — true: `[ ]`, predicted: `[ ]` (confidence `[ ]`)
> Post: `[paste]`
> Analysis: `[...]`

**Example 3** — true: `[ ]`, predicted: `[ ]` (confidence `[ ]`)
> Post: `[paste]`
> Analysis: `[...]`

### 5.3 Sample classifications

> **⬜ FILL IN — 3–5 posts run through the fine-tuned model**, each with predicted
> label and confidence. For **at least one correct** prediction, add a sentence on
> why it's reasonable. (Written as text, not a screenshot.)

| # | Post (excerpt) | Predicted | Confidence | Correct? | Note |
|---|----------------|-----------|------------|----------|------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |

### 5.4 Reflection — what the model captured vs. what I intended

> **⬜ FILL IN.** This is *not* a list of wrong predictions — it's a higher-level
> observation about the gap between your label *definitions* and the decision
> boundary the model actually learned. What did it overfit to (e.g. post length,
> question marks, specific vocabulary)? What did it miss? Where does its boundary
> sit relative to the one you defined in `planning.md` §2–3?

---

## 6. Spec reflection

> **⬜ FILL IN / edit.** One way the spec/plan helped, and one way you diverged.
> Starting points based on this project (verify and make them your own):
> - **Helped:** Deciding the `critique`/`observation` scope rule and the §6 numeric
>   success criteria *before* annotating gave an objective bar to judge results
>   against, rather than rationalizing whatever the model produced.
> - **Diverged:** `[e.g. planned ~65/label but inquiry came in under target, so I
>   did targeted collection from new/rising — describe what actually happened]`.

## 7. AI usage

At least two specific instances (edit to reflect what *you* directed and overrode):

1. **Aligning the baseline prompt to the label spec.** Directed Claude Code to
   rewrite the notebook's Section 5 Groq system prompt to use the exact `planning.md`
   §2 definitions plus the §3 tiebreak rules and a strict "label name only" output
   instruction. It produced the revised prompt cell. *What I changed/overrode:*
   `[fill in — e.g. tweaked an example, adjusted wording]`.
2. **Tooling for collection and reporting.** Directed Claude Code to build
   `collect_posts.py` (public-Reddit collector → annotation scaffold) and
   `make_report.py`, and to extend the notebook so `evaluation_results.json`
   includes per-class metrics and confusion matrices. *What I changed/overrode:*
   `[fill in — e.g. adjusted target count, reviewed/edited script]`.
3. **Annotation pre-labeling (if used) — disclose or remove.** `[If you used an LLM
   to pre-label before reviewing, name the tool and how many rows; the
   prelabeled_by_ai / human_corrected columns track which. If not used, say so.]`
4. **Failure-pattern surfacing.** `[Describe the §5.2 step: what you asked the LLM
   to find in the wrong predictions and what you verified or discarded.]`

## 8. Demo video

> **⬜ FILL IN — link to your 3–5 min demo.** Show 3–5 posts classified by the
> fine-tuned model with label + confidence visible, narrate one correct and one
> incorrect prediction, and walk through this evaluation report.

---

## 9. Reproducing

1. Open the notebook in Colab (badge at the top) and select a **T4 GPU** runtime.
2. Add Colab Secrets: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`,
   `REDDIT_USER_AGENT`, `GROQ_API_KEY`. **Never commit the Groq key.**
3. Collect + annotate: `python3 collect_posts.py` (or the notebook's PRAW cell),
   label the rows, save as `data/labeled_dataset.csv`.
4. Run Sections 1–6 to split, fine-tune, evaluate, and run the baseline.
5. Download `evaluation_results.json` and `confusion_matrix.png`, commit them.
6. `python3 make_report.py` → paste the tables into §5.
