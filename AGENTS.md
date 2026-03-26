# AGENTS.md - Mood Machine Project Guide

This document provides essential information for AI coding agents working on the Mood Machine project.

## Project Overview

The Mood Machine is an educational text classification project designed to teach fundamental concepts of sentiment analysis and machine learning through hands-on experimentation. The project implements **two parallel approaches** to mood classification:

1. **Rule-Based Model** (`mood_analyzer.py`): A lexicon-based classifier using predefined positive/negative word lists
2. **ML Model** (`ml_experiments.py`): A scikit-learn classifier using bag-of-words features and logistic regression

The primary goal is educational: learners implement missing functionality, experiment with data, compare approaches, and document findings in a model card.

## Technology Stack

- **Language**: Python 3.13+
- **Core Dependencies**:
  - `scikit-learn` - Machine learning library for the ML model
  - `matplotlib` - Visualization (optional for extended experiments)
  - `ipykernel` - Jupyter notebook support (optional)
- **Virtual Environment**: `.venv` directory (pre-configured)

## Project Structure

```
.
├── AGENTS.md              # This file - guidance for AI agents
├── README.md              # Human-facing project documentation
├── model_card.md          # Template for documenting model findings
├── requirements.txt       # Python dependencies
├── dataset.py             # Shared data: word lists and labeled examples
├── mood_analyzer.py       # Rule-based classifier (has TODOs to implement)
├── ml_experiments.py      # ML classifier implementation (complete)
└── main.py                # Entry point - runs evaluation and interactive demo
```

### Module Responsibilities

| File | Purpose | Status |
|------|---------|--------|
| `dataset.py` | Defines `POSITIVE_WORDS`, `NEGATIVE_WORDS`, `SAMPLE_POSTS`, and `TRUE_LABELS` | Needs expansion (TODO present) |
| `mood_analyzer.py` | `MoodAnalyzer` class with `preprocess()`, `score_text()`, `predict_label()`, and `explain()` methods | Has TODOs to implement |
| `ml_experiments.py` | ML training pipeline using scikit-learn | Complete, functional |
| `main.py` | Evaluation functions and interactive demo runner | Complete, functional |

## Build and Run Commands

### Environment Setup

```bash
# Activate the virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Or on Unix/macOS
source .venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### Running the Project

```bash
# Run the rule-based model (main entry point)
python main.py

# Run the ML experiments
python ml_experiments.py
```

### What Happens When You Run `main.py`

1. **Evaluation Phase**: Tests the rule-based model against `SAMPLE_POSTS` and reports accuracy
2. **Batch Demo**: Shows predictions for all sample posts
3. **Interactive Mode**: Prompts for user input to test custom sentences

## Code Organization and Architecture

### Data Flow

```
dataset.py
    ↓
mood_analyzer.py  ←→  ml_experiments.py
    ↓                      ↓
main.py (evaluation)   ml_experiments.py (training & evaluation)
```

### Label System

The project supports four mood labels:
- `"positive"` - Expresses positive sentiment
- `"negative"` - Expresses negative sentiment
- `"neutral"` - No strong sentiment
- `"mixed"` - Contains both positive and negative elements

Labels are defined in `TRUE_LABELS` (in `dataset.py`) and must align with `SAMPLE_POSTS` entries.

### Rule-Based Model Architecture

```python
class MoodAnalyzer:
    - __init__(positive_words, negative_words)  # Uses dataset.py defaults
    - preprocess(text) → List[str]              # Tokenization (TODO: improve)
    - score_text(text) → int                    # Calculate sentiment score (TODO: implement)
    - predict_label(text) → str                 # Map score to label (TODO: implement)
    - explain(text) → str                       # Show reasoning (partially implemented)
```

### ML Model Architecture

```python
# Feature extraction
CountVectorizer() → Bag-of-words representation

# Classification
LogisticRegression(max_iter=1000) → Multi-class classifier
```

## Key Implementation Tasks (TODOs)

The following tasks are explicitly marked as TODOs in the codebase:

### In `dataset.py`:
- Add 5-10 more posts to `SAMPLE_POSTS` with matching `TRUE_LABELS`
- Include diverse language: slang, emojis, sarcasm, mixed emotions

### In `mood_analyzer.py`:
1. **Preprocessing Improvements**: Remove punctuation, handle emojis, normalize repeated characters ("soooo" → "soo")
2. **Scoring Logic**: Implement word counting, handle negation ("not happy"), weight different words, handle emojis/slang
3. **Label Prediction**: Map scores to labels (may adjust thresholds for "mixed" category)
4. **Explanation Enhancement**: Show which words contributed to the score

## Development Conventions

### Coding Style

- **Type Hints**: All functions use Python type annotations (e.g., `def func(text: str) -> int:`)
- **Docstrings**: Google-style docstrings explaining parameters, returns, and behavior
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Comments**: Extensive inline comments and TODO markers for educational clarity

### File Patterns

- No complex build system - direct Python execution
- No test suite - evaluation is done against labeled dataset
- No linting configuration - keep code simple and readable

## Testing Strategy

There is no formal test suite. Instead, the project uses:

1. **Labeled Dataset Evaluation**: `evaluate_rule_based()` and `evaluate_on_dataset()` functions compare predictions against `TRUE_LABELS`
2. **Interactive Testing**: Users can type custom sentences to observe model behavior
3. **Model Card Documentation**: Learners document strengths, weaknesses, and failure cases

### Expected Workflow for Development

1. Implement TODOs in `mood_analyzer.py`
2. Run `python main.py` to see evaluation results
3. Add more data to `dataset.py` to test edge cases
4. Run `python ml_experiments.py` to compare approaches
5. Document findings in `model_card.md`

## Common Failure Modes to Consider

When implementing improvements, learners should test against these known challenges:

- **Sarcasm**: "I absolutely love getting stuck in traffic"
- **Negation**: "I am not happy about this" (currently in dataset)
- **Mixed emotions**: "Feeling tired but kind of hopeful" (currently in dataset)
- **Emojis**: ":)", ":(", "🥲", "😂", "💀"
- **Slang**: "lowkey", "highkey", "no cap"
- **Repeated letters**: "soooo happy", "terribleee"

## Dependencies and External Resources

- **scikit-learn**: Used for `CountVectorizer`, `LogisticRegression`, and `accuracy_score`
- **matplotlib**: Available for visualization extensions (not used in starter code)
- **Standard Library Only**: Rule-based model uses only Python standard library

## Security Considerations

- No network requests or external APIs
- No file I/O beyond Python imports
- User input is only used for interactive prediction (no persistence)
- Safe for educational environments

## Agent Guidelines

When assisting with this project:

1. **Preserve the TODO Structure**: Don't remove TODO comments - they guide learners
2. **Maintain Educational Value**: Keep code readable with comments
3. **Follow Existing Patterns**: Use type hints and docstring style from existing code
4. **Test Both Models**: When suggesting changes, consider impact on both rule-based and ML approaches
5. **Align Data**: Ensure `SAMPLE_POSTS` and `TRUE_LABELS` always have matching lengths
6. **Suggest Edge Cases**: Help identify challenging examples for testing
