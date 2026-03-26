# Model Card: Mood Machine

This model card documents the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule-based model** (`mood_analyzer.py`)
2. A **machine learning model** (`ml_experiments.py`) using scikit-learn

---

## 1. Model Overview

**Model type:**  
Both models were implemented and compared.

**Intended purpose:**  
Classify short text posts (social media messages, chat snippets) into one of four mood labels: `positive`, `negative`, `neutral`, or `mixed`.

**How it works (brief):**

- **Rule-based**: Text is preprocessed (punctuation stripped, ASCII/Unicode emojis converted to sentiment tokens, repeated characters normalized). Each token is scored against positive (+1) and negative (−1) word lists. *Intensifier words* (`so`, `very`, `absolutely`, etc.) double the score of the next sentiment word. *Negator words* (`not`, `never`, `don't`, etc.) flip the sign of the next sentiment word, with a **2-word window** so phrases like `"not at all happy"` are handled correctly. The numeric score maps to a label: `positive`, `negative`, `neutral`, or `mixed` (when both word types cancel out).

- **ML model**: Each post is converted to a feature vector — either bag-of-words (`CountVectorizer`) or weighted bag-of-words (`TfidfVectorizer`). A `LogisticRegression` classifier is trained on those vectors and `TRUE_LABELS`.

---

## 2. Data

**Dataset description:**  
14 posts in `SAMPLE_POSTS` / `TRUE_LABELS` (training and development) and 6 posts in `TEST_POSTS` / `TEST_LABELS` (held-out, never seen during development). The original starter had 6 posts; 8 more were added covering slang, emojis, sarcasm, and mixed emotions.

**Labeling process:**  
Labels were chosen by reading each post and assigning the most natural human interpretation. Posts expressing only one sentiment type were labeled `positive` or `negative`. Posts where both sentiments clearly coexist were labeled `mixed`. Posts with no strong sentiment were labeled `neutral`.

Hardest-to-label posts:
- *"Lowkey stressed but highkey proud of myself"* — labeled `mixed`, but could reasonably be `positive`.
- *"I absolutely love getting stuck in traffic"* — sarcasm makes this `negative`, but the model sees `love` and predicts `positive`.
- *"No cap this is the best day ever"* — labeled `positive`, but `best`/`ever` are not in the lexicon.

**Important characteristics of the dataset:**
- Contains slang: `lowkey`, `highkey`, `no cap`, `lit`, `fire`, `mid`, `sus`, `trash`
- Includes ASCII emojis (`:)`, `:(`) and Unicode emoji (💀)
- One clear sarcasm example (instructive failure case)
- Several mixed-feeling posts
- Short, ambiguous messages ("This is fine")

**Possible issues:**
- Very small (14 training posts) — insufficient for reliable ML training
- Slight label imbalance: 5 positive, 4 negative, 1 neutral, 4 mixed
- Sarcasm requires world knowledge neither model has
- Slang terms outside the lexicon are invisible to the rule-based scorer

---

## 3. How the Rule-Based Model Works

**Scoring rules:**

| Feature | Description |
|---------|-------------|
| Positive words | Each match → +1 |
| Negative words | Each match → −1 |
| **Intensifiers** | `so`, `very`, `absolutely`, etc. → next sentiment word scores ×2 |
| Negation | `not`, `never`, `don't`, etc. → flips sign of next sentiment word |
| **Extended negation window** | Window survives up to 2 non-sentiment tokens (e.g., `"not at all happy"` → −2) |
| Emoji (ASCII) | `:)` / `:D` → `__emoji_positive__` (+2); `:(` → `__emoji_negative__` (−2) |
| Emoji (Unicode) | Detected by Unicode codepoint; scored ±2 |
| Repeated chars | `soooo` → `soo` before lexicon lookup |
| Punctuation | Stripped so `"great!"` matches `"great"` |
| Mixed label | Score == 0 but both word types present → `"mixed"` |

**Strengths:**
- Handles negation across a 2-word window ("not at all happy" → negative)
- Intensifiers increase sensitivity ("so stressed" → −2 instead of −1)
- Transparent: `explain()` shows exactly which words contributed, including negated ones
- Works on any new text immediately — no training required
- Emoji and emoticon support adds signal not available in bag-of-words

**Weaknesses:**
- Cannot detect sarcasm ("I love getting stuck in traffic" → incorrectly positive)
- Blind to slang not in the lexicon (`best`, `ever`, `no cap`)
- Intensifiers don't carry over across clauses
- No intensity gradation for different negative words (hate ≈ annoyed = −1)

---

## 4. How the ML Model Works

**Features used:**  
Two representations compared side-by-side:
- `CountVectorizer` — raw word counts
- `TfidfVectorizer` — word counts weighted by how unique each word is across all posts

**Training data:**  
`SAMPLE_POSTS` (14 posts) with labels from `TRUE_LABELS`.

**Evaluation setup:**  
To avoid inflated accuracy from training on test data, both vectorizers are now evaluated on a separate held-out `TEST_POSTS` / `TEST_LABELS` split (6 posts).

**Results:**

| Vectorizer | Train acc | Test acc (held-out) |
|------------|-----------|---------------------|
| CountVectorizer | 1.00 | 0.50 |
| TF-IDF | 0.93 | 0.50 |

TF-IDF's lower training accuracy (0.93 vs 1.00) signals less overfitting — a more honest result. Both score 0.50 on the test set, consistent with having too few training examples for 4 classes.

**Strengths and weaknesses:**

| | Strength | Weakness |
|-|----------|----------|
| ML model | Learns patterns automatically; handles any vocabulary seen in training | Overfits with small data; not transparent; needs a real held-out test set for honest evaluation |

---

## 5. Evaluation

**How models were evaluated:**  
Both models evaluated on all 14 labeled posts in `SAMPLE_POSTS` (development accuracy) and 6 held-out posts in `TEST_POSTS` (test accuracy).

**Summary:**

| Model | Dev accuracy | Test accuracy |
|-------|-------------|---------------|
| Rule-based | 0.71 (10/14) | TBD — depends on additions to `TEST_LABELS` |
| CountVectorizer ML | 1.00 (14/14, overfitted) | 0.50 (3/6) |
| TF-IDF ML | 0.93 (13/14) | 0.50 (3/6) |

**Examples of correct predictions (rule-based):**

| Post | Label | Why correct |
|------|-------|-------------|
| "I am not happy about this" | negative | Negation: `not` + `happy` → −1 |
| "That was absolutely trash" | negative | Intensifier: `absolutely` + `trash` → −2 |
| "Just got my grades back :)" | positive | ASCII emoji `:)` → +2 |

**Examples of incorrect predictions (rule-based):**

| Post | Predicted | True | Why wrong |
|------|-----------|------|-----------|
| "I absolutely love getting stuck in traffic" | positive | negative | Sarcasm — model sees `absolutely` + `love` → +2 |
| "No cap this is the best day ever" | neutral | positive | `best`/`ever` not in lexicon |
| "Finally done with finals but I am so exhausted" | negative | mixed | Only `exhausted` found; no positive word to balance |

---

## 6. Limitations

- **Very small dataset**: 14 training posts is far too few for reliable ML training or evaluation
- **No sarcasm detection**: Rule-based model is entirely fooled by irony; ML model only handles it if trained examples cover that pattern
- **Lexicon coverage gap**: Slang not in `POSITIVE_WORDS` / `NEGATIVE_WORDS` is invisible to the rule-based scorer (e.g., `best`, `ever`, `no cap`)
- **Short negation scope**: The 2-word window helps but still misses longer constructions like `"not even close to being happy"`
- **Intensifier collateral**: Intensifiers apply to the immediately next sentiment word only; `"lowkey really stressed and sad"` would only double `stressed`, not `sad`
- **Label subjectivity**: Sarcasm and mixed-feeling posts may have multiple valid labels, creating noisy training signal
- **Language diversity**: Models not tested on non-English text, heavy AAVE, or regional slang

---

## 7. Ethical Considerations

- **Distress misclassification**: Sarcasm about distress ("I'm totally fine 🙂") could be classified as positive, masking a real need for support. High-stakes applications (mental health, crisis detection) require far higher reliability.
- **Cultural and linguistic bias**: Slang, dialects, and non-standard spellings are underrepresented. The model may systematically mis-classify posts from communities whose language isn't in the lexicon.
- **Privacy**: Processing personal messages raises consent and data-handling concerns, even in educational tools.
- **Overconfidence**: The ML model's 1.00 training accuracy could mislead users. Always report test-set accuracy.
- **Feedback loops**: If a mood-detection system acts on predictions (e.g., content moderation), systematic misclassification could disproportionately affect certain groups.

---

## 8. Ideas for Improvement

| # | Idea | Status |
|---|------|--------|
| 1 | Expand word lists with slang and intensifiers | ✅ Implemented |
| 2 | Add a real held-out test set | ✅ Implemented (`TEST_POSTS` / `TEST_LABELS`) |
| 3 | TF-IDF instead of CountVectorizer | ✅ Implemented (side-by-side comparison) |
| 4 | Extended negation window (2-word budget) | ✅ Implemented |
| 5 | Sarcasm detection | ⏳ Requires external sarcasm corpus |
| 6 | Word embeddings (GloVe / fastText) | ⏳ Requires large file downloads |
| 7 | Transformer model (distilBERT) | ⏳ Requires PyTorch + GPU/CPU compute |
| 8 | Larger / more balanced dataset | ⏳ Ongoing — add more posts to `SAMPLE_POSTS` |
