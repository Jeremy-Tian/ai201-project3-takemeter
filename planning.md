# TakeMeter — Project Planning

**AI201 · Project 3 — Fine-Tuning a Text Classifier**

> Draft derived from the starter notebook. Verify the dataset numbers and fill in
> the bracketed `[...]` fields once your annotation and Colab run are complete.

## 1. Problem statement

Posts in the r/TrueFilm community vary widely in intent: some are long-form
critical essays, some are narrow technical notes, and some exist mainly to start
a discussion. TakeMeter classifies each post into one of three intent categories
so the community could, for example, filter or route posts by type.

The task is a single-label, three-class text classification problem. We compare a
**zero-shot LLM baseline** (Groq `llama-3.3-70b-versatile`) against a
**fine-tuned `distilbert-base-uncased`** model trained on our own labeled data.

## 2. Label taxonomy

| Label | id | Definition |
|-------|----|-----------|
| `critique` | 0 | Long-form, structured analysis of a film's narrative, historical context, or overall artistic merit. |
| `observation` | 1 | A specific, focused comment on a technical element, a single scene, or an actor's performance. |
| `inquiry` | 2 | A post primarily intended to ask a question or start a debate about a director, genre, or specific film. |

**Boundary examples used during annotation:**
- `critique` — *"The use of color in Ki-duk's later works serves as a spiritual bridge rather than just a visual aesthetic."*
- `observation` — *"I noticed that the blocking in the dinner scene mimics a stage play to heighten the tension."*
- `inquiry` — *"Why do you think the French New Wave still resonates with modern independent filmmakers?"*

## 3. Data collection

- **Source:** r/TrueFilm "hot" posts, fetched via the Reddit API (PRAW).
- **Scope:** up to 250 posts; only posts with body text (`selftext`) are kept.
- **Raw output:** `truefilm_raw.csv` (title, text, id) — produced by the notebook.
- **Annotation:** each post is manually assigned one of the three labels above,
  producing the labeled dataset CSV (`text`, `label`) committed to this repo.

**Dataset size:** [N] labeled examples
**Label distribution:** critique [n], observation [n], inquiry [n]

## 4. Annotation guidelines

- Assign the label that matches the post's **primary intent**, not incidental content
  (a post that asks a question *and* offers some analysis is `inquiry` if the question
  is the point).
- When a post mixes structured argument with a single technical note, prefer
  `critique` if the analysis spans the whole work, `observation` if it is scoped to
  one scene/element.
- Edge cases and tie-breaking decisions are recorded alongside the data so labeling
  stays consistent.

## 5. Modeling approach

**Baseline — zero-shot (Groq `llama-3.3-70b-versatile`):**
- System prompt defines the three labels with one example each and instructs the
  model to return only the label name.
- `temperature=0`, `max_tokens=20`. Unparseable responses are excluded from scoring.

**Fine-tuned — `distilbert-base-uncased`:**
- 70 / 15 / 15 stratified train / validation / test split (`random_state=42`).
- Tokenized at `max_length=256`.
- Hyperparameters: 3 epochs, learning rate `2e-5`, train batch size 16,
  weight decay `0.01`, 50 warmup steps. Best model by validation accuracy is kept.

## 6. Evaluation plan

- Both models are scored on the **same locked test set**.
- Primary metric: **accuracy**; also report per-class precision/recall/F1 and a
  confusion matrix for the fine-tuned model.
- Error analysis: review misclassified test examples and discuss 3 in depth
  (see README evaluation report).
- Deliverables exported from Colab: `evaluation_results.json`, `confusion_matrix.png`.

## 7. Hypothesis

Fine-tuning on in-domain, hand-labeled data is expected to outperform the zero-shot
baseline, especially on the `critique` vs `observation` boundary where the
distinction depends on scope rather than surface vocabulary. Actual results are
recorded in the README once the Colab run is complete.
