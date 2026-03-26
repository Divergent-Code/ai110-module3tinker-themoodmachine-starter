# Edge Case Testing Report: Mood Machine

**Date:** 2026-03-25  
**Test Suite:** 61 adversarial examples across 10 categories  
**Models Tested:** Rule-based classifier vs. ML (TF-IDF + Logistic Regression)

---

## Executive Summary

This report documents the results of adversarial testing designed to expose weaknesses in two sentiment classification approaches: a rule-based lexicon classifier and a machine learning model. Both models struggle with nuanced language phenomena, but in different ways.

| Model | Overall Accuracy | Key Strength | Key Weakness |
|-------|-----------------|--------------|--------------|
| **Rule-Based** | 21.3% (13/61) | Negation handling, Mixed emotion detection | Sarcasm, Context-dependent words |
| **ML Model** | 41.0% (25/61) | Sarcasm detection, Implicit sentiment | Negation, Mixed emotions |

**Key Insight:** Neither model achieves majority accuracy on adversarial examples. The ML model outperforms overall, but the rule-based model wins in specific categories where its handcrafted logic applies.

---

## Methodology

### Test Categories

We designed 61 test cases across 10 challenging categories:

1. **Sarcasm & Irony** (7 tests) - Positive words used in negative contexts
2. **Complex Negation** (7 tests) - Double negatives, extended negation scope
3. **Context-Dependent Words** (7 tests) - Slang with inverted meanings ("sick", "bad")
4. **But-Clause Override** (5 tests) - "X but Y" where Y should dominate
5. **Emoji Misinterpretation** (6 tests) - Unicode emojis with ambiguous sentiment
6. **Subtle Mixed Emotions** (5 tests) - Balanced positive and negative expressions
7. **Morphological Variations** (6 tests) - Word forms not in lexicon ("happiness", "unhappy")
8. **Implicit Sentiment** (6 tests) - No explicit sentiment words
9. **Intensifier Scope** (5 tests) - Complex intensifier + negation interactions
10. **Evolving Slang** (7 tests) - Modern idioms and phrases

### Evaluation Criteria

Each test case was labeled with the "true" sentiment based on human judgment. A prediction was marked correct if it matched any acceptable label (e.g., "neutral/negative" accepted either).

---

## Results by Category

### 1. Sarcasm & Irony

| Metric | Rule-Based | ML Model |
|--------|-----------|----------|
| Accuracy | **0.0%** (0/7) | **71.4%** (5/7) |

**Analysis:** The rule-based model has no mechanism to detect incongruity between positive words and negative context. It scores "I absolutely love getting stuck in traffic" as **positive (+2)** because "love" gets +1 and "absolutely" doubles it.

The ML model, trained on limited data, surprisingly catches some sarcasm patterns, likely learning from the single sarcastic example in the training set ("I absolutely love getting stuck in traffic" is in SAMPLE_POSTS).

**Example Failures:**
```
Input:    "Oh great, another meeting 🙃"
Expected: negative
Rule:     positive (score: 1)     ← "great" in positive word list
ML:       negative                ← Learned pattern

Input:    "Fantastic, the WiFi is down again"
Expected: negative
Rule:     positive (score: 1)     ← "fantastic" in positive word list
ML:       negative                ← Contextual learning
```

**Key Finding:** Sarcasm detection requires understanding context, speaker intent, and real-world knowledge—none of which simple models possess.

---

### 2. Complex Negation

| Metric | Rule-Based | ML Model |
|--------|-----------|----------|
| Accuracy | **28.6%** (2/7) | **14.3%** (1/7) |

**Analysis:** The rule-based model implements a 2-word negation window that can skip non-sentiment tokens. This works for simple cases but breaks with double negations or when the window is exceeded.

**Example Success (Rule-Based):**
```
Input:    "Never have I been so happy"
Expected: positive
Rule:     positive (score: 2)     ← "Never" + "happy" with intensifier
ML:       negative                ← Misinterprets "Never" as dominant
```

**Example Failures:**
```
Input:    "You can't say I'm not happy"
Expected: positive
Rule:     negative (score: -1)    ← Double negation confuses window
ML:       negative                ← Bag-of-words loses negation structure

Input:    "not only good but amazing"
Expected: positive
Rule:     neutral (score: 0)      ← "not" negates "good" (-1), "amazing" (+1)
ML:       negative                ← "not" outweighs positives
```

**Key Finding:** Negation scope is linguistically complex. The rule-based model's fixed window approach handles simple cases better than the ML model's bag-of-words, but both fail on complex constructions.

---

### 3. Context-Dependent Words

| Metric | Rule-Based | ML Model |
|--------|-----------|----------|
| Accuracy | **0.0%** (0/7) | **42.9%** (3/7) |

**Analysis:** Words like "sick", "bad", "wicked" can mean "excellent" in slang contexts. The rule-based model cannot adapt word meanings based on context.

**Example Failures (Rule-Based):**
```
Input:    "This song is sick!"
Expected: positive (slang)
Rule:     neutral                 ← "sick" not in any word list

Input:    "This is bad!"
Expected: positive (slang)
Rule:     negative (score: -1)    ← "bad" in negative word list
```

**Key Finding:** Static lexicons cannot handle polysemous words or evolving slang. The ML model's distributed representations capture some contextual variation.

---

### 4. But-Clause Override

| Metric | Rule-Based | ML Model |
|--------|-----------|----------|
| Accuracy | **0.0%** (0/5) | **40.0%** (2/5) |

**Analysis:** Humans interpret "I'm tired but feeling amazing" as positive because "but" signals that the second clause overrides the first. Neither model has this pragmatic knowledge.

**Example Failures:**
```
Input:    "I'm tired but feeling amazing"
Expected: positive
Rule:     mixed (score: 0)        ← tired (-1) + amazing (+1) = 0
ML:       mixed                   ← Similar averaging

Input:    "Stressed about work but grateful for my team"
Expected: positive
Rule:     mixed (score: 0)        ← stressed (-1) + grateful (+1) = 0
ML:       negative                ← Stress words dominate
```

**Key Finding:** Discourse markers like "but" carry critical sentiment-shifting information. Simple models treat all words as additive rather than understanding discourse structure.

---

### 5. Emoji Misinterpretation

| Metric | Rule-Based | ML Model |
|--------|-----------|----------|
| Accuracy | **16.7%** (1/6) | **33.3%** (2/6) |

**Analysis:** The rule-based model maps ASCII emojis (":)", ":(") to sentiment sentinels but passes Unicode emojis through unprocessed. Many emojis have context-dependent meanings.

**Example Failures:**
```
Input:    "That's just great 🥲"
Expected: negative (pained smile)
Rule:     positive (score: 1)     ← "great" = +1, 🥲 unmapped
ML:       positive                ← "great" dominates

Input:    "😂 I'm crying"
Expected: positive (amused)
Rule:     neutral                 ← 😂 unmapped, "crying" not in negative list
ML:       negative                ← Associates "crying" with sadness

Input:    "😭😭 finally done"
Expected: positive (relief)
Rule:     neutral                 ← 😭 unmapped
ML:       negative                ← Crying emojis = sadness
```

**Key Finding:** Emoji sentiment is highly context-dependent. 🥲 (smiling through pain), 😂 (joy), and 😭 (can be relief) require contextual interpretation beyond simple mapping.

---

### 6. Subtle Mixed Emotions

| Metric | Rule-Based | ML Model |
|--------|-----------|----------|
| Accuracy | **60.0%** (3/5) | **0.0%** (0/5) |

**Analysis:** The rule-based model detects "mixed" when positive and negative words both present but cancel to score=0. The ML model must explicitly learn this category.

**Example Success (Rule-Based):**
```
Input:    "Happy for them, sad for me"
Expected: mixed
Rule:     mixed (score: 0)        ← happy (+1) + sad (-1) = 0, both present
ML:       negative                ← Picks dominant sentiment
```

**Example Failures:**
```
Input:    "Excited about the trip, dreading the flight"
Expected: mixed
Rule:     positive (score: 1)     ← excited (+1), "dreading" not in list
ML:       positive                ← "excited" dominates
```

**Key Finding:** The rule-based model's explicit mixed detection (score=0 with both polarities) works well for balanced expressions. The ML model lacks this explicit mechanism.

---

### 7. Morphological Variations

| Metric | Rule-Based | ML Model |
|--------|-----------|----------|
| Accuracy | **16.7%** (1/6) | **50.0%** (3/6) |

**Analysis:** The rule-based model uses exact string matching. Words like "happiness" (noun), "unhappy" (negated adjective), and "loving" (participle) are not in the lexicon.

**Example Failures (Rule-Based):**
```
Input:    "happiness"
Expected: positive
Rule:     neutral                 ← "happiness" ≠ "happy"

Input:    "unhappy"
Expected: negative
Rule:     neutral                 ← "unhappy" not in negative word list

Input:    "loving this"
Expected: positive
Rule:     neutral                 ← "loving" ≠ "love"
```

**Key Finding:** Stemming or lemmatization would improve the rule-based model. The ML model's TF-IDF vectorizer captures some morphological similarity through character n-grams.

---

### 8. Implicit Sentiment

| Metric | Rule-Based | ML Model |
|--------|-----------|----------|
| Accuracy | **0.0%** (0/6) | **50.0%** (3/6) |

**Analysis:** Some expressions carry sentiment without explicit sentiment words. "I got the job!" implies success; "Lost my keys again" implies frustration.

**Example Failures (Rule-Based):**
```
Input:    "I got the job!"
Expected: positive
Rule:     neutral                 ← No sentiment words in text

Input:    "Lost my keys again"
Expected: negative
Rule:     neutral                 ← "Lost" not in negative word list
```

**Key Finding:** World knowledge is required for implicit sentiment. The ML model picks up some associations from training data (e.g., "won" → positive).

---

### 9. Intensifier Scope

| Metric | Rule-Based | ML Model |
|--------|-----------|----------|
| Accuracy | **80.0%** (4/5) | **80.0%** (4/5) |

**Analysis:** Both models handle simple intensifiers well ("absolutely terrible" = strong negative). Complex interactions with negation are challenging.

**Example Successes:**
```
Input:    "so not happy"
Expected: negative
Rule:     negative (score: -1)    ← "so" intensifies "happy", negation flips
ML:       negative                ← Learned pattern

Input:    "not very happy"
Expected: negative
Rule:     negative (score: -2)    ← "very" doubles, negation flips
ML:       negative                ← Learned pattern
```

**Example Failure:**
```
Input:    "literally the worst"
Expected: negative
Rule:     neutral                 ← "worst" not in word list
ML:       positive                ← "literally" may associate with positive
```

**Key Finding:** Intensifiers work when the target word is in the lexicon. Coverage gaps ("worst") hurt both models.

---

### 10. Evolving Slang

| Metric | Rule-Based | ML Model |
|--------|-----------|----------|
| Accuracy | **28.6%** (2/7) | **28.6%** (2/7) |

**Analysis:** Modern slang phrases like "rent free in my head" (can't stop thinking about) or "ate and left no crumbs" (did an excellent job) require cultural knowledge.

**Example Failures:**
```
Input:    "highkey obsessed with this"
Expected: positive
Rule:     neutral                 ← "obsessed" not in positive list
ML:       negative                ← "obsessed" → negative association

Input:    "rent free in my head"
Expected: positive
Rule:     neutral                 ← No sentiment words
ML:       negative                ← "free" context misinterpreted
```

**Key Finding:** Slang evolves rapidly. Static word lists and small training datasets cannot keep pace with linguistic innovation.

---

## Comparative Analysis

### Where Rule-Based Wins

1. **Explicit Mixed Detection:** Score-zero logic correctly identifies balanced emotions
2. **Negation with Known Words:** 2-word window handles simple negation well
3. **Interpretability:** Can explain exactly which words contributed to the score

### Where ML Wins

1. **Sarcasm Detection:** Learned patterns from training data
2. **Context Adaptation:** Distributed representations capture some word similarity
3. **Implicit Sentiment:** Associates events with sentiment from training

### Where Both Fail

1. **Discourse Structure:** Neither understands "but" clause override
2. **Complex Negation:** Double negatives confuse both
3. **Unicode Emojis:** Context-dependent emoji meanings
4. **World Knowledge:** Requires understanding of situations, not just words

---

## Recommendations

### For the Rule-Based Model

1. **Add Stemming/Lemmatization:** Handle "happiness" → "happy", "loving" → "love"
2. **Expand Emoji Mapping:** Add sentiment mappings for Unicode emojis
3. **But-Clause Detection:** Special handling for "X but Y" patterns
4. **Negation Improvements:** Handle double negatives, extend window selectively

### For the ML Model

1. **More Training Data:** Current 14 examples insufficient for generalization
2. **Better Features:** Use n-grams to capture negation patterns ("not good")
3. **Emoji Embeddings:** Include emoji2vec or similar representations
4. **Class Balancing:** Ensure "mixed" examples are properly represented

### For Both Models

1. **Active Learning:** Collect failures and add to training data
2. **Ensemble Methods:** Combine rule-based and ML predictions
3. **Confidence Scoring:** Flag low-confidence predictions for human review

---

## Conclusion

This adversarial testing reveals fundamental limitations in simple sentiment classification approaches:

- **Lexicon-based models** fail when language deviates from their word lists or when context changes word meanings.

- **Bag-of-words ML models** lose critical word order information and struggle with negation, despite better generalization.

- **Both approaches** lack the world knowledge, discourse understanding, and pragmatic reasoning that humans use for sentiment interpretation.

The ML model's 41% vs. 21% accuracy advantage demonstrates the value of learning from data, but both scores are far below production-ready thresholds. This highlights why modern sentiment analysis uses deep learning (transformers, contextual embeddings) that better capture linguistic nuance.

For educational purposes, this comparison effectively demonstrates the trade-offs between interpretability (rule-based) and generalization (ML), as well as the challenges of NLP that seem simple to humans but are difficult for machines.

---

## Appendix: Full Test Results

See `test_edge_cases.py` for the complete test suite and reproduction instructions.

```bash
# Run the tests
python test_edge_cases.py
```

### Test Data Summary
- **Total Cases:** 61
- **Categories:** 10
- **Rule-Based Correct:** 13/61 (21.3%)
- **ML Correct:** 25/61 (41.0%)
- **Both Correct:** 6/61 (9.8%)
- **Both Wrong:** 29/61 (47.5%)
- **Disagreement:** 26/61 (42.6%)
