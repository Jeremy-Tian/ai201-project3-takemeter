#!/usr/bin/env python3
"""Collect public r/TrueFilm posts into a CSV scaffold for manual annotation.

This uses Reddit's PUBLIC JSON endpoints (no login, no API key) so everything
collected is public content. It pulls across several sortings to get enough
variety and volume to reach 200+ labeled examples with reasonable class balance.

It does NOT assign labels — it leaves the `label` and `notes` columns empty for
you to fill in by reading each post. (If you want to pre-label with an LLM and
then review, that's a separate optional step; see planning.md §7.2.)

Usage:
    python3 collect_posts.py                  # ~300 posts -> data/truefilm_raw.csv
    python3 collect_posts.py --target 350 --out data/truefilm_raw.csv

If Reddit rate-limits or blocks public JSON from your network, fall back to the
notebook's PRAW scraping cell (which uses your Reddit API credentials in Colab).
"""

import argparse
import csv
import json
import os
import time
import urllib.request
import urllib.error

SUBREDDIT = "TrueFilm"
# (sorting, query params) — spread across sortings so "inquiry"-type posts that
# rarely reach "hot" still show up (see planning.md §4 underrepresentation plan).
LISTINGS = [
    ("hot", ""),
    ("top", "t=year"),
    ("top", "t=month"),
    ("new", ""),
    ("rising", ""),
]
USER_AGENT = "takemeter-ai201:collect:0.1 (course project, manual annotation)"
FIELDS = ["id", "title", "text", "label", "notes",
          "prelabeled_by_ai", "human_corrected", "score", "url"]


def fetch_listing(sort, params, after=None):
    url = f"https://www.reddit.com/r/{SUBREDDIT}/{sort}.json?limit=100&raw_json=1"
    if params:
        url += "&" + params
    if after:
        url += "&after=" + after
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)["data"]


def collect(target):
    seen = {}
    for sort, params in LISTINGS:
        after = None
        for _ in range(10):  # up to ~1000 per sorting via pagination
            if len(seen) >= target:
                break
            try:
                data = fetch_listing(sort, params, after)
            except urllib.error.HTTPError as e:
                print(f"  [{sort} {params}] HTTP {e.code} — skipping rest of this sorting")
                break
            except Exception as e:
                print(f"  [{sort} {params}] error: {e} — skipping")
                break

            children = data.get("children", [])
            new_here = 0
            for c in children:
                p = c["data"]
                body = (p.get("selftext") or "").strip()
                if not body or p["id"] in seen:
                    continue
                seen[p["id"]] = {
                    "id": p["id"],
                    "title": p.get("title", "").strip(),
                    "text": body,
                    "label": "",
                    "notes": "",
                    "prelabeled_by_ai": "",
                    "human_corrected": "",
                    "score": p.get("score", ""),
                    "url": "https://www.reddit.com" + p.get("permalink", ""),
                }
                new_here += 1
            print(f"  [{sort} {params or '-'}] +{new_here} new (total {len(seen)})")

            after = data.get("after")
            if not after:
                break
            time.sleep(1.5)  # be polite to the public endpoint
        if len(seen) >= target:
            break
    return list(seen.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=300,
                    help="number of unique posts to aim for (default 300)")
    ap.add_argument("--out", default="data/truefilm_raw.csv",
                    help="output CSV path (default data/truefilm_raw.csv)")
    args = ap.parse_args()

    print(f"Collecting up to {args.target} public r/{SUBREDDIT} posts with body text...")
    rows = collect(args.target)
    if not rows:
        print("\nNo posts collected. Reddit may be blocking public JSON from your "
              "network — use the notebook's PRAW scraping cell instead.")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} posts to {args.out}")
    print("Next: open it in a spreadsheet and fill the `label` column by reading "
          "each post (definitions in planning.md §2). Log tricky cases in `notes`.")
    print("When done, save the labeled file as data/labeled_dataset.csv and commit it.")


if __name__ == "__main__":
    main()
