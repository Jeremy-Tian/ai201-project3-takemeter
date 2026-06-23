# TakeMeter — Project Planning

**AI201 · Project 3 — Fine-Tuning a Text Classifier**

TakeMeter classifies r/TrueFilm posts by their *intent* into three categories,
comparing a zero-shot LLM baseline against a fine-tuned DistilBERT model. This
document records the design decisions made **before** annotation begins.

---

## 1. Community

**Chosen community: [r/TrueFilm](https://www.reddit.com/r/TrueFilm/).**

r/TrueFilm is a subreddit for serious, long-form discussion of cinema — it
explicitly discourages low-effort posts, so nearly every submission has real body
text rather than just a link or a meme. That matters for a text classifier: there
is actual language to learn from in almost every example.

It is a good fit for a classification task because the discourse is **varied in
intent but consistent in domain**. Posts range from multi-paragraph critical
essays, to narrow technical notes about a single scene, to open questions meant to
start a debate. The *subject matter* (film) is shared across all of them, so the
classifier can't cheat by keying on topic vocabulary — it has to learn the
**rhetorical shape** of each post type. That makes the three labels genuinely hard
to separate on surface features alone, which is what makes the task interesting
rather than trivial.

A practical motivation: a tool that auto-tags posts by intent could let the
community filter (e.g. "show me only discussion questions today") or help
moderators route long critiques differently from quick observations.

---

## 2. Labels

Three single-intent labels. Each post is assigned the label matching its
**primary purpose**, not its incidental content.

### `critique`
A long-form, structured argument that analyzes a film's narrative, themes,
historical context, or overall artistic merit as a whole work.

- *"Across his late period, Kim Ki-duk uses color not as decoration but as a moral register — the shift to muted palettes tracks each protagonist's loss of innocence, and reading the films chronologically makes that arc unmistakable."*
- *"Tarkovsky's long takes aren't slow for their own sake; they force the viewer into the film's own sense of time, which is why summarizing the plot of Stalker tells you nothing about what the film actually is."*

### `observation`
A focused comment on a single technical element, one specific scene, or an
individual performance, rather than the work as a whole.

- *"I only noticed on rewatch that the blocking in the dinner scene mirrors a stage play — everyone stays on their side of the table until the accusation lands."*
- *"The match cut from the bone to the spacecraft is famous, but the sound edit underneath it is what actually sells the jump."*

### `inquiry`
A post whose primary purpose is to ask a question or prompt community discussion
about a director, genre, film, or idea.

- *"Why do you think the French New Wave still resonates with modern independent filmmakers, when so many other movements feel dated?"*
- *"What's a film you initially disliked but came to admire after a second viewing, and what changed for you?"*

---

## 3. Hard edge cases

The genuinely ambiguous boundary is **`critique` vs `observation`**. Both analyze
the film closely; the difference is *scope*. A post can start as a narrow note
about one scene and then generalize into a claim about the whole film, sitting
right on the line.

A second, lower-frequency ambiguity is **`inquiry` vs `critique`**: a post that
makes a substantial argument but ends with "...or am I wrong?" — is the point the
argument (critique) or the invitation to debate (inquiry)?

**Handling rule (decided up front, applied consistently):**
- **Scope wins for critique/observation.** If the analysis is anchored to a single
  scene, shot, performance, or technical choice, it is `observation` — even if it's
  insightful. If it makes a claim about the work *as a whole*, it is `critique`.
- **Primary intent wins for inquiry.** If removing the question still leaves a
  complete, self-standing argument, it's `critique`; the question is rhetorical. If
  removing the question leaves nothing substantive, it's `inquiry`.
- Every tie-breaking decision is logged in an `annotation_notes` column (or a
  running notes file) so the same call is made every time the pattern recurs.

---

## 4. Data collection plan

- **Source:** r/TrueFilm "hot" posts via the Reddit API (PRAW), keeping only posts
  that have body text (`selftext`). The notebook scrapes up to 250 posts into
  `truefilm_raw.csv`, which is then annotated by hand.
- **Target volume:** **200+ labeled examples total**, aiming for a roughly balanced
  split of **~65–70 per label**. Perfect balance isn't required, but no label
  should fall below ~50 examples, because the 15% test split (~30 examples) needs
  enough of each class to produce meaningful per-class metrics.
- **If a label is underrepresented after 200 examples:** I will do **targeted
  collection** rather than stopping at 200. `inquiry` is the most likely to be
  scarce in "hot" (questions get fewer upvotes than essays), so I'll supplement by
  pulling from r/TrueFilm's "new" and "rising" sortings and from the weekly
  discussion threads, and by searching the subreddit for question-shaped titles.
  I will collect until the smallest class has at least ~50 examples, then re-balance
  by trimming the largest class if needed. The final per-label counts are recorded
  in this document and the README.

**Final dataset stats (filled in after annotation):**
total `[N]` · critique `[n]` · observation `[n]` · inquiry `[n]`

---

## 5. Evaluation metrics

Both models are scored on the **same locked 15% test set**. Accuracy alone is
insufficient here for two reasons: the classes may be imbalanced (so a model can
look good by favoring the majority class), and the three labels are not equally
easy — the `critique`/`observation` boundary is hard, and overall accuracy hides
*where* a model fails.

Metrics used and why:

- **Overall accuracy** — headline number and the primary baseline-vs-fine-tuned
  comparison.
- **Per-class precision, recall, and F1** — the core of the analysis. These reveal
  whether the model handles the rare/hard classes or just the easy majority. I
  expect `critique` recall to be the telling number, since `critique` posts are the
  ones most likely to be misread as `observation`.
- **Macro-averaged F1** — averages F1 across classes *without* weighting by class
  size, so a model can't hide poor performance on a small class behind a large one.
  This is the single fairest summary number for an imbalanced 3-class task.
- **Confusion matrix** — shows the *direction* of errors (e.g. how often `critique`
  is mistaken for `observation` vs the reverse), which directly drives the error
  analysis and any label-definition tightening.

Reported for both the Groq baseline and the fine-tuned model so improvements can be
attributed per class, not just in aggregate.

---

## 6. Definition of success

**Genuinely useful (the target):** the fine-tuned model reaches **≥ 0.80 overall
accuracy** and **≥ 0.75 macro-F1**, with **no single class below 0.65 F1**, *and*
beats the zero-shot baseline by a **margin larger than the noise** of a ~30-example
test set. At that level the auto-tagging is reliable enough that a reader filtering
by intent would trust it most of the time.

**Good enough for deployment in a real community tool:** **≥ 0.75 accuracy** and
**every class above 0.60 F1**, *provided* the dominant error mode is the
"forgivable" `critique`↔`observation` confusion (two analytical labels) rather than
confusing `inquiry` with either — mislabeling a discussion question as an essay
would be the most user-visible failure, so `inquiry` recall specifically must be
**≥ 0.70**.

**Not good enough:** any class below 0.60 F1, `inquiry` recall below 0.70, or a
fine-tuned model that fails to clearly beat the baseline (which would mean the
labeled data added no signal a general LLM didn't already have).

### Are these criteria objective?

Yes — each threshold is a specific number computed directly from
`evaluation_results.json` and the per-class report the notebook prints, so at the
end I can mechanically check each condition (overall accuracy, macro-F1, per-class
F1 floor, `inquiry` recall, and baseline margin) and state pass/fail without
judgment calls. The one qualitative criterion (error mode must be the
`critique`↔`observation` confusion) is read directly off the confusion matrix.

---

## 7. AI Tool Plan

This is an annotation-and-evaluation project, not an implementation project — there
is essentially no application code to generate. AI tools are used at the three
points where they genuinely help:

### 7.1 Label stress-testing (before annotating)
I will give an LLM (Claude) my three label definitions from §2 and the edge-case
description from §3, and ask it to generate **5–10 posts that deliberately sit on
the `critique`/`observation` and `inquiry`/`critique` boundaries**. If I can't
classify those generated posts cleanly using my handling rules, the definitions are
too loose — I will tighten §2/§3 **before** annotating 200 real examples, so I'm not
re-deciding the same ambiguity 200 times. The tightened definitions and any rule
changes are recorded here.

### 7.2 Annotation assistance (during annotation)
I **will** use an LLM to **pre-label** the raw posts in a first pass, then review
every pre-label myself and correct it — the model proposes, I decide. Tracking for
disclosure:
- Tool: Claude (or the Groq `llama-3.3-70b-versatile` baseline prompt itself).
- A `prelabeled_by_ai` boolean column marks every example the model labeled first,
  and a `human_corrected` column marks where my review changed the label. This lets
  me report exactly how many labels were AI-suggested vs human-originated in the AI
  usage section, and lets me check whether AI pre-labeling biased the dataset toward
  the baseline model's preferences.
- The locked test set will be **annotated by hand without AI pre-labeling** to keep
  the evaluation independent of the baseline model.

### 7.3 Failure analysis (after evaluation)
After running the notebook, I will take the list of **wrong predictions** the
notebook prints (text, true label, predicted label, confidence) and give it to an
LLM, asking it to identify **patterns** in the errors — e.g. "are long posts
systematically misread as critique?", "do low-confidence errors cluster on one
class boundary?", "is there shared vocabulary in the misclassified posts?".
I will **verify each proposed pattern myself** by going back to the actual
misclassified examples and confirming the pattern holds (and isn't the model
pattern-matching on too few cases) before it goes into the README evaluation report.
The three in-depth error cases in the report will be ones I've personally
re-read, not taken on the AI's word.

> This plan is updated before any stretch features are attempted later.

---

## 8. Modeling approach (reference)

For completeness — the modeling pipeline is provided by the starter notebook:

- **Baseline:** zero-shot Groq `llama-3.3-70b-versatile`, `temperature=0`,
  prompted with the §2 label definitions; unparseable responses excluded from
  scoring.
- **Fine-tuned:** `distilbert-base-uncased`, 70/15/15 stratified split
  (`random_state=42`), `max_length=256`, 3 epochs, lr `2e-5`, train batch size 16,
  weight decay `0.01`, 50 warmup steps; best model selected by validation accuracy.
- Deliverables exported from Colab: `evaluation_results.json`, `confusion_matrix.png`.
