"""
load_goEmotions.py
==================
One-time script that downloads a sample of Google's GoEmotions dataset
(https://github.com/google-research/google-research/tree/master/goemotions)
and injects balanced training examples into dataset.py.

Usage:
    python load_goEmotions.py

What it does:
    1. Downloads the GoEmotions train CSV directly from GitHub (no pip needed).
    2. Maps the 28 GoEmotions labels to our 4-label system.
    3. Samples up to MAX_PER_LABEL examples per label (balanced).
    4. Deduplicates against existing SAMPLE_POSTS and TEST_POSTS.
    5. Appends the new posts as a clearly-marked block in dataset.py.

Label mapping (28 → 4):
    positive : admiration, amusement, approval, caring, desire,
               excitement, gratitude, joy, love, optimism, pride, relief
    negative : anger, annoyance, disappointment, disapproval, disgust,
               embarrassment, fear, grief, nervousness, remorse, sadness
    neutral  : neutral
    mixed    : confusion, curiosity, realization, surprise
               (and any comment with BOTH positive + negative labels)
"""

import csv
import io
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Number of examples per label to add (balanced).
MAX_PER_LABEL = 20

# Seed for reproducibility.
RANDOM_SEED = 42

# GoEmotions train split — raw CSV on GitHub.
GOEMOTIONS_CSV_URL = (
    "https://raw.githubusercontent.com/google-research/google-research"
    "/master/goemotions/data/full_dataset/goemotions_1.csv"
)

# Emotion → our 4-label mapping.
_POSITIVE_EMOTIONS = {
    "admiration", "amusement", "approval", "caring",
    "desire", "excitement", "gratitude", "joy",
    "love", "optimism", "pride", "relief",
}
_NEGATIVE_EMOTIONS = {
    "anger", "annoyance", "disappointment", "disapproval",
    "disgust", "embarrassment", "fear", "grief",
    "nervousness", "remorse", "sadness",
}
_NEUTRAL_EMOTIONS = {"neutral"}
_MIXED_EMOTIONS   = {"confusion", "curiosity", "realization", "surprise"}

# Path to this project's dataset.py (same directory as this script).
DATASET_PATH = Path(__file__).parent / "dataset.py"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_label(emotion_names: list[str]) -> str | None:
    """
    Map a list of GoEmotions emotion names to our 4-label scheme.

    Returns None if the emotion set is ambiguous or unmapped.
    """
    pos = any(e in _POSITIVE_EMOTIONS for e in emotion_names)
    neg = any(e in _NEGATIVE_EMOTIONS for e in emotion_names)
    neu = any(e in _NEUTRAL_EMOTIONS  for e in emotion_names)
    mix = any(e in _MIXED_EMOTIONS    for e in emotion_names)

    if pos and neg:
        return "mixed"
    if pos:
        return "positive"
    if neg:
        return "negative"
    if neu:
        return "neutral"
    if mix:
        return "mixed"
    return None  # unknown emotion — skip


def _load_existing_posts() -> set[str]:
    """Return a lowercase set of all posts already in dataset.py."""
    from dataset import SAMPLE_POSTS, TEST_POSTS
    return {p.lower().strip() for p in SAMPLE_POSTS + TEST_POSTS}


def _clean_text(text: str) -> str:
    """Light cleaning: strip surrounding whitespace and quotes."""
    return text.strip().strip('"').strip("'").strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Downloading GoEmotions CSV...")
    try:
        with urllib.request.urlopen(GOEMOTIONS_CSV_URL, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as exc:
        print(f"ERROR: Could not download GoEmotions CSV.\n  {exc}")
        print("Check your internet connection and try again.")
        return

    print("Parsing CSV and mapping labels...")

    # GoEmotions CSV columns: text, emotions (comma-separated), id
    reader = csv.reader(io.StringIO(raw))
    header = next(reader)  # skip header

    # Collect rows by mapped label.
    buckets: dict[str, list[str]] = defaultdict(list)
    skipped = 0

    for row in reader:
        if len(row) < 2:
            continue
        text_raw = row[0]
        emotions_raw = row[1]  # comma-separated emotion names

        text = _clean_text(text_raw)

        # Skip very short or very long texts (keep it similar to SAMPLE_POSTS).
        if len(text) < 5 or len(text) > 200:
            skipped += 1
            continue

        # Skip texts with URLs or Reddit artifacts.
        if any(token in text.lower() for token in ["http", "[removed]", "[deleted]"]):
            skipped += 1
            continue

        emotion_names = [e.strip() for e in emotions_raw.split(",") if e.strip()]
        label = _map_label(emotion_names)
        if label is None:
            skipped += 1
            continue

        buckets[label].append(text)

    print(f"  Rows skipped: {skipped}")
    for lbl, rows in buckets.items():
        print(f"  {lbl:>10}: {len(rows):>5} candidates")

    # Sample balanced subset, avoiding duplicates with existing posts.
    existing = _load_existing_posts()
    rng = random.Random(RANDOM_SEED)

    selected_posts: list[str] = []
    selected_labels: list[str] = []

    for label in ("positive", "negative", "neutral", "mixed"):
        candidates = [t for t in buckets[label] if t.lower() not in existing]
        rng.shuffle(candidates)
        chosen = candidates[:MAX_PER_LABEL]
        selected_posts.extend(chosen)
        selected_labels.extend([label] * len(chosen))
        print(f"  Selected {len(chosen):>3} {label} examples")

    if not selected_posts:
        print("No new examples found. dataset.py is unchanged.")
        return

    # Build the Python block to append to dataset.py.
    posts_block = "\n# --- GoEmotions data (auto-loaded by load_goEmotions.py) ---\n"
    for text, label in zip(selected_posts, selected_labels):
        # Escape any single quotes in the text.
        safe_text = text.replace("\\", "\\\\").replace("'", "\\'")
        posts_block += f"SAMPLE_POSTS.append('{safe_text}')\n"
        posts_block += f"TRUE_LABELS.append('{label}')\n"

    # Append to dataset.py.
    current = DATASET_PATH.read_text(encoding="utf-8")
    if "GoEmotions data (auto-loaded" in current:
        print(
            "WARNING: GoEmotions block already present in dataset.py.\n"
            "  Delete the existing block before re-running this script."
        )
        return

    DATASET_PATH.write_text(current + posts_block, encoding="utf-8")

    print(f"\nDone! Added {len(selected_posts)} examples to dataset.py.")
    print(f"SAMPLE_POSTS now has {14 + len(selected_posts)} entries.")  # approximate
    print("\nNext steps:")
    print("  python -c \"from dataset import SAMPLE_POSTS,TRUE_LABELS; "
          "assert len(SAMPLE_POSTS)==len(TRUE_LABELS); "
          "print('lengths OK:', len(SAMPLE_POSTS))\"")
    print("  python ml_experiments.py")


if __name__ == "__main__":
    main()
