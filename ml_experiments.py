# ml_experiments.py
"""
Simple ML experiments for the Mood Machine lab.

This file uses a "real" machine learning library (scikit-learn)
to train a tiny text classifier on the same SAMPLE_POSTS and
TRUE_LABELS that you use with the rule based model.

#8 Improvements applied:
  - TF-IDF vectorizer added as an alternative to CountVectorizer.
  - Held-out TEST_POSTS / TEST_LABELS used for a fairer accuracy estimate.
"""

from typing import List, Tuple, Union

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from dataset import SAMPLE_POSTS, TRUE_LABELS, TEST_POSTS, TEST_LABELS

# Type alias so function signatures work with either vectorizer.
Vectorizer = Union[CountVectorizer, TfidfVectorizer]


def train_ml_model(
    texts: List[str],
    labels: List[str],
    vectorizer: Vectorizer = None,
) -> Tuple[Vectorizer, LogisticRegression]:
    """
    Train a simple text classifier using bag of words features
    and logistic regression.

    Steps:
      1. Convert the texts into numeric vectors using the given vectorizer
         (defaults to CountVectorizer; pass TfidfVectorizer for TF-IDF).
      2. Fit a LogisticRegression model on those vectors and labels.

    Args:
        texts: List of text strings to train on.
        labels: Corresponding mood labels.
        vectorizer: A fitted CountVectorizer or TfidfVectorizer instance.
                    If None, a new CountVectorizer is created.

    Returns:
        (vectorizer, model)
    """
    if len(texts) != len(labels):
        raise ValueError(
            "texts and labels must be the same length. "
            "Check SAMPLE_POSTS and TRUE_LABELS in dataset.py."
        )

    if not texts:
        raise ValueError("No training data provided. Add examples in dataset.py.")

    if vectorizer is None:
        vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    return vectorizer, model


def evaluate_on_dataset(
    texts: List[str],
    labels: List[str],
    vectorizer: Vectorizer,
    model: LogisticRegression,
) -> float:
    """
    Evaluate the trained model on a labeled dataset.

    Prints each text with its predicted label and the true label,
    then returns the overall accuracy as a float between 0 and 1.
    """
    if len(texts) != len(labels):
        raise ValueError(
            "texts and labels must be the same length. "
            "Check your dataset."
        )

    X = vectorizer.transform(texts)
    preds = model.predict(X)

    correct = 0
    for text, true_label, pred_label in zip(texts, labels, preds):
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        print(f'  "{text}" -> predicted={pred_label}, true={true_label}')

    accuracy = accuracy_score(labels, preds)
    print(f"  Accuracy: {accuracy:.2f} ({correct}/{len(labels)})")
    return accuracy


def predict_single_text(
    text: str,
    vectorizer: Vectorizer,
    model: LogisticRegression,
) -> str:
    """
    Predict the mood label for a single text string using
    the trained ML model.
    """
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return pred


def run_interactive_loop(
    vectorizer: Vectorizer,
    model: LogisticRegression,
) -> None:
    """
    Let the user type their own sentences and see the ML model's
    predicted mood label.

    Type 'quit' or press Enter on an empty line to exit.
    """
    print("\n=== Interactive Mood Machine (ML model) ===")
    print("Type a sentence to analyze its mood.")
    print("Type 'quit' or press Enter on an empty line to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input == "" or user_input.lower() == "quit":
            print("Goodbye from the ML Mood Machine.")
            break

        label = predict_single_text(user_input, vectorizer, model)
        print(f"ML model: {label}")


if __name__ == "__main__":
    print("Training ML models on SAMPLE_POSTS and TRUE_LABELS from dataset.py...")
    print("Make sure you have added enough labeled examples before running this.\n")

    # ── Side-by-side: CountVectorizer vs TF-IDF (#8 improvement) ───────────
    results = {}
    for name, vec in [("CountVectorizer", CountVectorizer()),
                      ("TF-IDF",          TfidfVectorizer())]:
        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")

        # Train on SAMPLE_POSTS.
        vec_fitted, clf = train_ml_model(SAMPLE_POSTS, TRUE_LABELS, vec)

        # Evaluate on training data (expected ~1.0 — overfitted on small set).
        print(f"  --- Training set ({len(SAMPLE_POSTS)} posts) ---")
        train_acc = evaluate_on_dataset(SAMPLE_POSTS, TRUE_LABELS, vec_fitted, clf)

        # Evaluate on held-out test set (fairer estimate).
        print(f"\n  --- Held-out test set ({len(TEST_POSTS)} posts) ---")
        test_acc = evaluate_on_dataset(TEST_POSTS, TEST_LABELS, vec_fitted, clf)

        results[name] = {"train": train_acc, "test": test_acc}

    # ── Comparison summary ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Vectorizer':<20} {'Train acc':>10} {'Test acc':>10}")
    print(f"  {'-'*42}")
    for name, acc in results.items():
        print(f"  {name:<20} {acc['train']:>10.2f} {acc['test']:>10.2f}")
    print()
    print("  Note: train accuracy is inflated (model trained on same data).")
    print("  Test accuracy on the held-out set is the fairer number.")

    # ── Interactive loop uses TF-IDF model ──────────────────────────────────
    print("\nStarting interactive demo with TF-IDF model...")
    tfidf_vec, tfidf_clf = train_ml_model(
        SAMPLE_POSTS, TRUE_LABELS, TfidfVectorizer()
    )
    run_interactive_loop(tfidf_vec, tfidf_clf)

    print("\nTip: Compare these predictions with the rule based model")
    print("by running `python main.py`. Notice where they fail in")
    print("similar ways and where they fail in different ways.")
