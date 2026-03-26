# AGENTS.md - Mood Machine Project Guide

This document provides essential information for AI coding agents working on the Mood Machine project.

## Project Overview

The Mood Machine is an educational text classification lab designed to teach fundamental concepts of sentiment analysis, natural language processing, and machine learning fairness. The project implements **two parallel approaches** to mood classification:

1. **Rule-Based Model** (`mood_analyzer.py`): A lexicon-based classifier using predefined positive/negative word lists, negation handling, intensifiers, and emoji support
2. **ML Model** (`ml_experiments.py`): A scikit-learn classifier using bag-of-words/TF-IDF features and logistic regression

The educational goal is to help learners understand: how basic classification systems work, where they break, how different modeling choices affect fairness and accuracy, and how to document model behavior through a model card.

## Technology Stack

- **Language**: Python 3.x
- **Core Dependencies** (from `requirements.txt`):
  - `scikit-learn` - Machine learning library for the ML model
  - `matplotlib` - Visualization utilities
  - `ipykernel` - Jupyter notebook support (optional)
- **Virtual Environment**: `.venv` directory (pre-configured)

## Project Structure

```
.
├── AGENTS.md              # This file - guidance for AI agents
├── README.md              # Human-facing project documentation
├── model_card.md          # Detailed model documentation with findings
├── Phase_1_Implementation_Plan.md  # Implementation planning document
├── requirements.txt       # Python dependencies
├── dataset.py             # Shared data: word lists, labeled examples, test set
├── mood_analyzer.py       # Rule-based classifier (fully implemented)
├── ml_experiments.py      # ML classifier implementation (complete)
└── main.py                # Entry point - runs evaluation and interactive demo
```

### Module Responsibilities

| File | Purpose |
|------|---------|
| `dataset.py` | Defines `POSITIVE_WORDS`, `NEGATIVE_WORDS`, `INTENSIFIER_WORDS`, `SAMPLE_POSTS` (14 posts), `TRUE_LABELS`, `TEST_POSTS` (6 posts), and `TEST_LABELS` |
| `mood_analyzer.py` | `MoodAnalyzer` class with full preprocessing, scoring, prediction, and explanation capabilities |
| `ml_experiments.py` | ML training pipeline with CountVectorizer and TfidfVectorizer comparison |
| `main.py` | Rule-based evaluation, batch demo, and interactive mode |

## Build and Run Commands

### Environment Setup

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Or on Unix/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

```bash
# Run the rule-based model (main entry point)
python main.py

# Run the ML experiments with side-by-side comparison
python ml_experiments.py
```

### What Happens When You Run `main.py`

1. **Evaluation Phase**: Tests the rule-based model against `SAMPLE_POSTS` with `TRUE_LABELS` and reports accuracy
2. **Batch Demo**: Shows predictions for all sample posts without labels
3. **Interactive Mode**: Prompts for user input to test custom sentences (type 'quit' or press Enter to exit)

### What Happens When You Run `ml_experiments.py`

1. Trains two models (CountVectorizer and TfidfVectorizer) on `SAMPLE_POSTS`
2. Evaluates each on training data and held-out `TEST_POSTS`
3. Displays comparison summary with train/test accuracy
4. Starts interactive mode using the TF-IDF model

## Code Organization and Architecture

### Data Flow

```
dataset.py (word lists + labeled data)
    ↓
mood_analyzer.py ←→ ml_experiments.py
    ↓                       ↓
main.py (rule-based)    (ML-based)
```

### Label System

The project supports four mood labels:
- `"positive"` - Score > 0 (expresses positive sentiment)
- `"negative"` - Score < 0 (expresses negative sentiment)
- `"neutral"` - Score == 0, no sentiment words found
- `"mixed"` - Score == 0, but both positive and negative words present

### Rule-Based Model Features

The `MoodAnalyzer` class implements:

| Feature | Description |
|---------|-------------|
| Preprocessing | ASCII/Unicode emoji handling, punctuation stripping, repeated character normalization |
| Scoring | Positive words (+1), negative words (-1), emoji sentinels (±2) |
| Negation | Flips sentiment of next word; 2-word window for phrases like "not at all happy" |
| Intensifiers | Doubles score of next sentiment word (so, very, absolutely, etc.) |
| Label Mapping | Score → positive/negative/neutral/mixed based on thresholds |
| Explanation | Shows contributing words, negated words, and final score |

### ML Model Features

- **Vectorizers**: `CountVectorizer` (raw counts) and `TfidfVectorizer` (weighted)
- **Classifier**: `LogisticRegression(max_iter=1000)`
- **Evaluation**: Train accuracy (on SAMPLE_POSTS) and test accuracy (on held-out TEST_POSTS)

## Dataset Structure

### Training/Development Set (`SAMPLE_POSTS` / `TRUE_LABELS`)
- 14 posts covering diverse language patterns
- Includes: slang (lowkey, highkey, no cap), emojis (ASCII and Unicode), sarcasm, negation, mixed emotions

### Held-out Test Set (`TEST_POSTS` / `TEST_LABELS`)
- 6 posts never used during development
- Used for fair evaluation of both models

### Word Lists (from `dataset.py`)
- `POSITIVE_WORDS`: 25 words including standard and slang terms (lit, fire, goat, vibing, valid)
- `NEGATIVE_WORDS`: 25 words including standard and slang terms (mid, sus, trash, cringe, dead)
- `INTENSIFIER_WORDS`: 13 words (so, very, really, absolutely, literally, lowkey, highkey, etc.)

## Development Conventions

### Coding Style

- **Type Hints**: All functions use Python type annotations (e.g., `def func(text: str) -> int:`)
- **Docstrings**: Google-style docstrings with Args/Returns sections
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for module-level constants
- **Constants**: Word lists are module-level constants; private module variables use `_leading_underscore`
- **Comments**: Extensive inline comments explaining implementation decisions

### Key Code Patterns

```python
# Type aliases for flexibility
Vectorizer = Union[CountVectorizer, TfidfVectorizer]

# Sentinel tokens for emojis
_ASCII_EMOJI_MAP: Dict[str, str] = {
    ":)": "__emoji_positive__",
    ":(": "__emoji_negative__",
    # ...
}

# Negator set for O(1) lookup
_NEGATORS = {"not", "never", "no", "cant", "can't", "dont", "don't", "wont", "won't"}
```

## Testing Strategy

There is no formal test suite. Evaluation is performed through:

1. **Labeled Dataset Evaluation**: 
   - `evaluate_rule_based()` in `main.py` computes accuracy against `TRUE_LABELS`
   - `evaluate_on_dataset()` in `ml_experiments.py` uses sklearn's `accuracy_score`

2. **Interactive Testing**: Both entry points provide interactive modes for manual testing

3. **Model Card Documentation**: Comprehensive documentation of findings, limitations, and ethical considerations in `model_card.md`

### Expected Accuracy (Current Implementation)

| Model | Dev Accuracy | Test Accuracy |
|-------|-------------|---------------|
| Rule-based | ~0.71 (10/14) | Varies by implementation |
| CountVectorizer ML | 1.00 (14/14) | 0.50 (3/6) |
| TF-IDF ML | 0.93 (13/14) | 0.50 (3/6) |

## Known Failure Modes and Edge Cases

The models struggle with these challenging cases (documented in model_card.md):

| Case | Example | Issue |
|------|---------|-------|
| Sarcasm | "I absolutely love getting stuck in traffic" | Model sees positive words, misses irony |
| Slang gaps | "No cap this is the best day ever" | "best"/"ever" not in lexicon |
| Mixed feelings | "Finally done with finals but I am so exhausted" | Only negative words detected |
| Short ambiguous | "This is fine" | Hard to classify even for humans |

## Dependencies and External Resources

- **scikit-learn**: Used for vectorizers, LogisticRegression, and metrics
- **matplotlib**: Available for visualization (not currently used in main scripts)
- **Standard Library Only**: Rule-based model uses `re`, `string`, `typing` only

## Security Considerations

- No network requests or external APIs
- No file I/O beyond Python imports
- User input is only used for interactive prediction (no persistence)
- Safe for educational environments

## Agent Guidelines

When assisting with this project:

1. **Preserve Documentation**: Keep docstrings and comments that explain educational concepts
2. **Maintain Data Alignment**: Ensure `SAMPLE_POSTS` and `TRUE_LABELS` always have matching lengths
3. **Follow Existing Patterns**: Use type hints and docstring style from existing code
4. **Test Both Models**: When suggesting changes, consider impact on both rule-based and ML approaches
5. **Consider Edge Cases**: Help identify challenging examples for testing (sarcasm, slang, emojis)
6. **Update Model Card**: If making significant changes, suggest corresponding updates to `model_card.md`
7. **Resist Over-engineering**: This is an educational project - keep solutions simple and explainable
