"""
Adversarial Edge Case Testing for Mood Machine

This script tests both the rule-based and ML models against challenging
text inputs designed to expose weaknesses in sentiment classification.

Run with: python test_edge_cases.py
"""

from mood_analyzer import MoodAnalyzer
from ml_experiments import train_ml_model, predict_single_text
from dataset import SAMPLE_POSTS, TRUE_LABELS
from typing import List, Tuple, Dict
import re
import sys
import io

# Force UTF-8 encoding for output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Initialize models
print("=" * 70)
print("INITIALIZING MODELS")
print("=" * 70)

rule_based = MoodAnalyzer()

# Train ML model on sample data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(SAMPLE_POSTS)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, TRUE_LABELS)

print("[OK] Rule-based model initialized")
print("[OK] ML model trained on SAMPLE_POSTS")
print()

# Define all edge case categories
EDGE_CASES: List[Tuple[str, str, str]] = [
    # Format: (category, test_input, expected_sentiment)
    
    # === CATEGORY 1: SARCASM & IRONY ===
    ("Sarcasm", "I absolutely love getting stuck in traffic", "negative"),
    ("Sarcasm", "Oh great, another meeting 🙃", "negative"),
    ("Sarcasm", "Yeah, because that's exactly what I needed today", "negative"),
    ("Sarcasm", "Fantastic, the WiFi is down again", "negative"),
    ("Sarcasm", "I really enjoy waiting 2 hours for a cold pizza", "negative"),
    ("Sarcasm", "Just what I needed, a flat tire", "negative"),
    ("Sarcasm", "Wonderful, my flight is delayed 5 hours", "negative"),
    
    # === CATEGORY 2: COMPLEX NEGATION ===
    ("Negation", "not only good but amazing", "positive"),
    ("Negation", "It's not that I hate it, I just don't love it", "neutral/negative"),
    ("Negation", "not the best experience", "negative"),
    ("Negation", "You can't say I'm not happy", "positive"),
    ("Negation", "It's not just good, it's incredible", "positive"),
    ("Negation", "I wouldn't say I'm unhappy", "positive"),
    ("Negation", "Never have I been so happy", "positive"),
    
    # === CATEGORY 3: CONTEXT-DEPENDENT WORDS ===
    ("Context", "This song is sick!", "positive"),
    ("Context", "That beat is absolutely killing me", "positive"),
    ("Context", "This is bad!", "positive"),  # slang meaning
    ("Context", "The movie was wicked", "positive"),
    ("Context", "That's a nasty move (in basketball)", "positive"),
    ("Context", "The drop on this song is dirty", "positive"),
    ("Context", "That outfit is criminal", "positive"),
    
    # === CATEGORY 4: BUT-CLAUSE OVERRIDE ===
    ("But-Clause", "I'm tired but feeling amazing", "positive"),
    ("But-Clause", "Stressed about work but grateful for my team", "positive"),
    ("But-Clause", "Angry at the situation but love the outcome", "positive"),
    ("But-Clause", "Terrible start but great ending", "positive"),
    ("But-Clause", "Had an awful morning but the day turned around", "positive"),
    
    # === CATEGORY 5: EMOJI MISINTERPRETATION ===
    ("Emoji", "That's just great 🥲", "negative"),
    ("Emoji", "😂 I'm crying", "positive"),
    ("Emoji", "😐 well okay then", "neutral"),
    ("Emoji", "😭😭 finally done", "positive"),
    ("Emoji", "Can't believe it 💀", "positive"),  # positive shock
    ("Emoji", "So beautiful I could cry 🥹", "positive"),
    
    # === CATEGORY 6: SUBTLE MIXED EMOTIONS ===
    ("Mixed", "Happy for them, sad for me", "mixed"),
    ("Mixed", "Love the food, hate the service", "mixed"),
    ("Mixed", "So proud and so exhausted", "mixed"),
    ("Mixed", "Excited about the trip, dreading the flight", "mixed"),
    ("Mixed", "Grateful it's over, wish it had gone better", "mixed"),
    
    # === CATEGORY 7: MORPHOLOGICAL VARIATIONS ===
    ("Morphology", "happiness", "positive"),
    ("Morphology", "unhappy", "negative"),
    ("Morphology", "loving this", "positive"),
    ("Morphology", "so stressed out", "negative"),
    ("Morphology", "happier than ever", "positive"),
    ("Morphology", "feeling hopeless", "negative"),
    
    # === CATEGORY 8: IMPLICIT SENTIMENT ===
    ("Implicit", "I got the job!", "positive"),
    ("Implicit", "They rejected my application", "negative"),
    ("Implicit", "We won!", "positive"),
    ("Implicit", "Lost my keys again", "negative"),
    ("Implicit", "Surprise!", "ambiguous"),
    ("Implicit", "OMG", "ambiguous"),
    
    # === CATEGORY 9: INTENSIFIER SCOPE ===
    ("Intensifier", "so not happy", "negative"),
    ("Intensifier", "very not good", "negative"),
    ("Intensifier", "absolutely terrible", "negative"),
    ("Intensifier", "literally the worst", "negative"),
    ("Intensifier", "not very happy", "negative"),
    
    # === CATEGORY 10: SLANG EVOLUTION ===
    ("Slang", "it's giving sad vibes", "negative"),
    ("Slang", "lowkey might delete later", "neutral"),
    ("Slang", "highkey obsessed with this", "positive"),
    ("Slang", "rent free in my head", "positive"),
    ("Slang", "understood the assignment", "positive"),
    ("Slang", "ate and left no crumbs", "positive"),
    ("Slang", "main character energy", "positive"),
]


def clean_expected(expected: str) -> List[str]:
    """Convert expected sentiment to list of acceptable labels."""
    if "/" in expected:
        return expected.split("/")
    return [expected]


def is_correct(predicted: str, expected: str) -> bool:
    """Check if prediction matches any acceptable expected label."""
    acceptable = clean_expected(expected)
    return predicted in acceptable


def run_tests():
    """Run all edge case tests and report results."""
    
    results: Dict[str, Dict] = {}
    total_tests = 0
    rule_correct = 0
    ml_correct = 0
    
    print("=" * 70)
    print("RUNNING EDGE CASE TESTS")
    print("=" * 70)
    print()
    
    current_category = None
    
    for category, text, expected in EDGE_CASES:
        # Print category header
        if category != current_category:
            current_category = category
            print()
            print("-" * 70)
            print(f"CATEGORY: {category}")
            print("-" * 70)
        
        # Get predictions
        rule_pred = rule_based.predict_label(text)
        rule_score = rule_based.score_text(text)
        ml_pred = predict_single_text(text, vectorizer, model)
        
        # Check correctness
        rule_ok = is_correct(rule_pred, expected)
        ml_ok = is_correct(ml_pred, expected)
        
        # Update counters
        total_tests += 1
        if rule_ok:
            rule_correct += 1
        if ml_ok:
            ml_correct += 1
        
        # Store results
        if category not in results:
            results[category] = {"total": 0, "rule_correct": 0, "ml_correct": 0, "tests": []}
        results[category]["total"] += 1
        if rule_ok:
            results[category]["rule_correct"] += 1
        if ml_ok:
            results[category]["ml_correct"] += 1
        results[category]["tests"].append({
            "text": text,
            "expected": expected,
            "rule_pred": rule_pred,
            "rule_score": rule_score,
            "ml_pred": ml_pred,
            "rule_ok": rule_ok,
            "ml_ok": ml_ok
        })
        
        # Print individual result
        rule_status = "[PASS]" if rule_ok else "[FAIL]"
        ml_status = "[PASS]" if ml_ok else "[FAIL]"
        
        print(f"\nText: \"{text}\"")
        print(f"  Expected: {expected}")
        print(f"  Rule-based: {rule_pred} (score: {rule_score}) {rule_status}")
        print(f"  ML model:   {ml_pred} {ml_status}")
    
    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY BY CATEGORY")
    print("=" * 70)
    print(f"{'Category':<20} {'Tests':<8} {'Rule-Based':<15} {'ML Model':<15}")
    print("-" * 70)
    
    for category in ["Sarcasm", "Negation", "Context", "But-Clause", "Emoji", 
                     "Mixed", "Morphology", "Implicit", "Intensifier", "Slang"]:
        if category in results:
            r = results[category]
            total = r["total"]
            rule_acc = r["rule_correct"] / total * 100
            ml_acc = r["ml_correct"] / total * 100
            print(f"{category:<20} {total:<8} {rule_acc:>5.1f}% ({r['rule_correct']}/{total}) {ml_acc:>5.1f}% ({r['ml_correct']}/{total})")
    
    print("-" * 70)
    print(f"{'OVERALL':<20} {total_tests:<8} {(rule_correct/total_tests*100):>5.1f}% ({rule_correct}/{total_tests}) {(ml_correct/total_tests*100):>5.1f}% ({ml_correct}/{total_tests})")
    print("=" * 70)
    
    # Detailed failure analysis
    print()
    print("=" * 70)
    print("FAILURE ANALYSIS")
    print("=" * 70)
    
    print("\nRule-Based Model Failures:")
    print("-" * 70)
    failures_shown = 0
    for category, data in results.items():
        for test in data["tests"]:
            if not test["rule_ok"]:
                print(f"  [{category}] \"{test['text']}\"")
                print(f"    Expected: {test['expected']}, Got: {test['rule_pred']} (score: {test['rule_score']})")
                failures_shown += 1
                if failures_shown >= 15:  # Limit output
                    print(f"    ... and {sum(1 for c in results.values() for t in c['tests'] if not t['rule_ok']) - 15} more")
                    break
        if failures_shown >= 15:
            break
    
    print("\nML Model Failures:")
    print("-" * 70)
    failures_shown = 0
    for category, data in results.items():
        for test in data["tests"]:
            if not test["ml_ok"]:
                print(f"  [{category}] \"{test['text']}\"")
                print(f"    Expected: {test['expected']}, Got: {test['ml_pred']}")
                failures_shown += 1
                if failures_shown >= 10:
                    print(f"    ... and {sum(1 for c in results.values() for t in c['tests'] if not t['ml_ok']) - 10} more")
                    break
        if failures_shown >= 10:
            break
    
    print()
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    # Find categories where rule-based beats ML and vice versa
    rule_wins = []
    ml_wins = []
    for category, data in results.items():
        if data["rule_correct"] > data["ml_correct"]:
            rule_wins.append(category)
        elif data["ml_correct"] > data["rule_correct"]:
            ml_wins.append(category)
    
    if rule_wins:
        print(f"\n✓ Rule-based wins in: {', '.join(rule_wins)}")
    if ml_wins:
        print(f"[WIN] ML model wins in: {', '.join(ml_wins)}")
    
    # Toughest categories
    sorted_by_difficulty = sorted(results.items(), key=lambda x: (x[1]["rule_correct"] + x[1]["ml_correct"]) / x[1]["total"])
    print(f"\nToughest categories (combined performance):")
    for cat, data in sorted_by_difficulty[:3]:
        combined = (data["rule_correct"] + data["ml_correct"]) / (data["total"] * 2) * 100
        print(f"  - {cat}: {combined:.1f}% combined accuracy")
    
    print()


if __name__ == "__main__":
    run_tests()
