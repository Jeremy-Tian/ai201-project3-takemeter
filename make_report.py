#!/usr/bin/env python3
"""Turn evaluation_results.json into paste-ready README markdown.

Run this AFTER you've downloaded evaluation_results.json from Colab and put it in
the repo root. It prints the results table, per-class metrics, and the confusion
matrix as a markdown table (Milestone 6) — copy the output into README.md.

    python3 make_report.py                 # reads ./evaluation_results.json
    python3 make_report.py path/to/results.json

Works with the enriched JSON the notebook now writes (per-class + confusion
matrices). If only the three accuracy fields are present, it prints what it can.
"""

import json
import sys


def pct(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)


def results_table(r):
    print("### Results\n")
    print("| Model | Accuracy |")
    print("|-------|----------|")
    print(f"| Zero-shot baseline (Groq) | {pct(r['baseline_accuracy'])} |")
    print(f"| Fine-tuned DistilBERT | {pct(r['finetuned_accuracy'])} |")
    print(f"| **Improvement** | {pct(r['improvement'])} |")
    print(f"\nTest set size: {r.get('test_set_size', '?')} examples.\n")


def per_class_table(report, title):
    if not report:
        return
    print(f"### Per-class metrics — {title}\n")
    print("| Label | Precision | Recall | F1 | Support |")
    print("|-------|-----------|--------|----|---------|")
    for label, m in report.items():
        if label in ("accuracy",):
            continue
        if not isinstance(m, dict):
            continue
        print(f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} "
              f"| {m['f1-score']:.3f} | {int(m['support'])} |")
    print()


def confusion_table(cm, labels, title):
    if not cm:
        return
    print(f"### Confusion matrix — {title}\n")
    print("Rows = true label, columns = predicted label.\n")
    header = "| true \\ pred | " + " | ".join(labels) + " |"
    sep = "|" + "---|" * (len(labels) + 1)
    print(header)
    print(sep)
    for i, row in enumerate(cm):
        print(f"| **{labels[i]}** | " + " | ".join(str(v) for v in row) + " |")
    print()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "evaluation_results.json"
    try:
        with open(path) as f:
            r = json.load(f)
    except FileNotFoundError:
        print(f"Not found: {path}\nDownload evaluation_results.json from Colab first.")
        sys.exit(1)

    labels = r.get("labels") or list((r.get("label_map") or {}).keys())

    results_table(r)
    per_class_table(r.get("finetuned_per_class"), "fine-tuned DistilBERT")
    confusion_table(r.get("finetuned_confusion_matrix"), labels, "fine-tuned DistilBERT")
    per_class_table(r.get("baseline_per_class"), "zero-shot baseline (Groq)")
    confusion_table(r.get("baseline_confusion_matrix"), labels, "zero-shot baseline (Groq)")


if __name__ == "__main__":
    main()
