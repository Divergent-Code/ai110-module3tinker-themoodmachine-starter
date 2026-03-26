# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

---

## 1. Model Overview

**Model type:**  
Both models were implemented and compared: a rule-based classifier and an ML-based classifier using logistic regression.

**Intended purpose:**  
Classify short text posts (social media messages, chat snippets) into one of four mood labels: `positive`, `negative`, `neutral`, or `mixed`.

**How it works (brief):**  
- **Rule-based**: Text is preprocessed (punctuation stripped, emojis converted to sentiment tokens, repeated characters normalized). Each token is checked against a positive-word list (+1) and a negative-word list (-1). Negation words (`not`, `never`, `don't`, etc.) flip the sign of the next sentiment word. The final numeric score maps to a label: positive, negative, neutral, or mixed (when both types of words appear and cancel out).
- **ML model**: Each post is converted to a bag-of-words vector using `CountVectorizer`. A `LogisticRegression` classifier is then trained on those vectors and the corresponding `TRUE_LABELS`.



## 2. Data

**Dataset description:**  
The dataset contains 14 posts in `SAMPLE_POSTS`, with matching labels in `TRUE_LABELS`. The original starter had 6 posts; 8 more were added to cover a wider variety of language styles.

**Labeling process:**  
Labels were chosen by reading each post and assigning the most natural human interpretation. Posts expressing only one sentiment type were labeled `positive` or `negative`. Posts where both sentiments clearly coexist were labeled `mixed`. Posts with no strong sentiment were labeled `neutral`.

The hardest posts to label:
- *"Lowkey stressed but highkey proud of myself"* — stress is negative, pride is positive; labeled `mixed`, but a human might reasonably call it `positive` overall.
- *"I absolutely love getting stuck in traffic"* — sarcasm makes this `negative`, but without context a reader might disagree.
- *"No cap this is the best day ever"* — labeled `positive`, but `best`/`ever` are not in the lexicon so the rule-based model cannot pick this up.

**Important characteristics of your dataset:**

- Contains slang (`lowkey`, `highkey`, `no cap`)
- Includes ASCII emojis (`:)`, `:(`) and Unicode emoji (💀)
- Includes one clear sarcasm example
- Several mixed-feeling posts
- Some short, ambiguous messages ("This is fine")

**Possible issues with the dataset:**  
- Very small (14 posts) — not nearly enough for reliable ML training or evaluation
- Slight imbalance: 5 positive, 4 negative, 1 neutral, 3 mixed (mixed is underrepresented)
- Sarcasm requires world knowledge the models don't have
- Slang terms (`no cap`, `lowkey`) are not in any lexicon, creating blind spots



## 3. How the Rule Based Model Works

**Scoring rules implemented:**

| Feature | Description |
|---------|-------------|
| Positive words | Each match → +1 to score |
| Negative words | Each match → -1 to score |
| Negation | `not`, `never`, `no`, `can't`, `don't`, `won't` flip the next word's sign |
| Emoji (ASCII) | `:)`, `:D`, `=)` → `__emoji_positive__` (+2); `:(`, `>:(`, `:/` → `__emoji_negative__` (-2) |
| Emoji (Unicode) | Detected by Unicode codepoint ranges; scored as +2 or -2 |
| Repeated chars | `soooo` → `soo` so the word still matches the lexicon |
| Punctuation | Stripped before lexicon lookup so `"great!"` still matches `"great"` |
| Mixed label | Score == 0 but both positive and negative words present → `"mixed"` |

**Strengths of this approach:**
- Handles negation correctly ("I am not happy" → negative)
- Transparent: `explain()` shows exactly which words contributed and why
- Works on any new text immediately — no training needed
- Emoji and ASCII emoticon support adds signal that bag-of-words misses semantically
- Generalises to unseen posts that use known vocabulary

**Weaknesses of this approach:**
- Cannot detect sarcasm ("I love getting stuck in traffic" → incorrectly positive)
- Depends on the size and quality of the word lists
- Slang terms not in the lexicon are invisible to the scorer
- Negation only covers one word ahead (e.g., "not at all bad" would miss "bad")
- Cannot detect intensity or emphasis beyond the ±2 emoji weight



## 4. How the ML Model Works

**Features used:**  
Bag-of-words representation using `CountVectorizer` — each post becomes a vector of word counts across the entire vocabulary.

**Training data:**  
Trained on all 14 posts in `SAMPLE_POSTS` with labels from `TRUE_LABELS`.

**Training behavior:**  
With only 14 training examples, the model easily memorises the training data. Adding more examples — especially diverse ones — is essential before drawing any conclusions from ML accuracy. When new posts were added to the dataset, the vocabulary grew and the model had more signal to work with, but accuracy on the training set stayed at 1.00 (since it can always memorise small sets).

**Strengths and weaknesses:**

| | Strength | Weakness |
|-|----------|---------|
| ML model | Learns patterns automatically; handles slang if seen in training | Overfits with small data; not transparent; needs a real test set |



## 5. Evaluation

**How the models were evaluated:**  
Both models were evaluated on all 14 labeled posts in `dataset.py` using exact-match label accuracy.

**Results:**

| Model | Accuracy | Correct / Total |
|-------|----------|-----------------|
| Rule-based | **0.71** | 10 / 14 |
| ML (LogisticRegression) | **1.00** | 14 / 14 ⚠️ overfitted |

> ⚠️ The ML model's 1.00 is inflated — it was evaluated on the same data it trained on.

**Examples of correct predictions (rule-based):**

| Post | Label | Why correct |
|------|-------|-------------|
| "I am not happy about this" | negative | Negation detected: `not` + `happy` → −1 |
| "Just got my grades back :)" | positive | ASCII emoji `:)` converted to `__emoji_positive__` → +2 |
| "Feeling tired but kind of hopeful" | mixed | `tired` (−1) + `hopeful` (+1) = 0, both present → mixed |

**Examples of incorrect predictions (rule-based):**

| Post | Predicted | True | Why wrong |
|------|-----------|------|-----------|
| "I absolutely love getting stuck in traffic" | positive | negative | Sarcasm — model sees `love` and scores +1 |
| "No cap this is the best day ever" | neutral | positive | `best` and `ever` not in `POSITIVE_WORDS` |
| "Finally done with finals but I am so exhausted" | negative | mixed | Only `exhausted` found; no balancing positive word |



## 6. Limitations

- **Very small dataset**: 14 posts is too few to train or meaningfully evaluate any ML model
- **No real test set**: Both models are evaluated on the data they were trained on or designed around
- **Sarcasm is unsolvable** for these approaches without context or a larger model
- **Lexicon coverage**: The rule-based model is blind to any word not in `POSITIVE_WORDS` or `NEGATIVE_WORDS`
- **No intensity scoring**: "I hate this" and "I'm slightly annoyed" get the same −1 score
- **Short negation window**: Only the immediately next sentiment word is flipped; "not even a little happy" would score "happy" correctly but miss more complex structures
- **Language diversity**: The models were not tested on non-English text, heavy AAVE, or other vernacular styles



## 7. Ethical Considerations

- **Distress misclassification**: A post expressing genuine distress through sarcasm or understated language could be classified as positive, masking a real need for support. Using mood detection for mental-health applications would require much higher reliability.
- **Cultural and linguistic bias**: Slang, dialects, and non-standard spellings (e.g., AAVE, internet-speak) are underrepresented in the word lists. The model may systematically mis-classify posts from certain communities.
- **Privacy**: Processing personal messages raises privacy concerns. Even an educational tool should not be applied to real user data without consent.
- **Label subjectivity**: Human labelers may disagree on edge cases (sarcasm, mixed emotions). Any mistakes in `TRUE_LABELS` directly impact both models.
- **Overconfidence**: The ML model's 1.00 training accuracy could mislead a user into trusting it more than it deserves.



## 8. Ideas for Improvement

- **Expand word lists**: Add more slang (`lit`, `fire`, `mid`, `sus`), intensifiers (`really`, `so`, `absolutely`), and domain-specific terms
- **Add a real held-out test set**: Keep some labeled examples for evaluation only, never used in development
- **TF-IDF instead of CountVectorizer**: Downweights common words and highlights more informative ones
- **Better negation**: Extend the negation window to 2–3 words (`not at all happy`)
- **Sarcasm detection**: Use irony cues or a separate classifier trained on sarcasm datasets
- **Word embeddings**: Use pre-trained vectors (e.g., GloVe, fastText) instead of bag-of-words so unseen slang near known words still gets useful representations
- **Transformer model**: A small pre-trained model (e.g., distilBERT fine-tuned on sentiment data) would handle context, sarcasm, and slang far better
- **Larger dataset**: Collect and label 100+ diverse posts before drawing conclusions from accuracy numbers
