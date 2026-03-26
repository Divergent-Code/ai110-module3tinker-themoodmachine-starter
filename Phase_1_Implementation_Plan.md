# Implement mood_analyzer.py TODOs

Fill in the four TODO methods in [mood_analyzer.py](file:///c:/Users/onika/OneDrive/Documents/Coding%20Projects/CodePath%20Projects/ai110-module3tinker-themoodmachine-starter/mood_analyzer.py) so the rule-based mood
classifier is fully functional. All changes preserve the existing TODO comments
and educational docstrings, per the AGENTS.md guidelines.

## Proposed Changes

### mood_analyzer.py

#### [MODIFY] [mood_analyzer.py](file:///c:/Users/onika/OneDrive/Documents/Coding%20Projects/CodePath%20Projects/ai110-module3tinker-themoodmachine-starter/mood_analyzer.py)

**[preprocess(text)](file:///c:/Users/onika/OneDrive/Documents/Coding%20Projects/CodePath%20Projects/ai110-module3tinker-themoodmachine-starter/mood_analyzer.py#39-59)** — improve tokenization:
- Strip punctuation from each token using `str.translate()`
- Detect and replace ASCII emojis (`:)`, `:(`, `>:|`, etc.) with sentiment tokens
  (`__emoji_positive__`, `__emoji_negative__`) before splitting
- Detect and replace Unicode emojis (😂, 🥲, 💀) with the same sentinel tokens
- Normalize repeated characters: collapse 3+ repeated letters to 2 (`soooo` → `soo`)

**[score_text(text)](file:///c:/Users/onika/OneDrive/Documents/Coding%20Projects/CodePath%20Projects/ai110-module3tinker-themoodmachine-starter/mood_analyzer.py#64-87)** — count sentiment words with negation:
- Call [preprocess()](file:///c:/Users/onika/OneDrive/Documents/Coding%20Projects/CodePath%20Projects/ai110-module3tinker-themoodmachine-starter/mood_analyzer.py#39-59) to get tokens
- Handle negation: if the previous token was a negator (`not`, `never`, `can't`,
  `dont`, `no`), flip the sentiment of the next word (+1 → -1 and vice versa)
- Score emoji sentinels: `__emoji_positive__` → +2, `__emoji_negative__` → -2
- Accumulate and return the integer score

**[predict_label(text)](file:///c:/Users/onika/OneDrive/Documents/Coding%20Projects/CodePath%20Projects/ai110-module3tinker-themoodmachine-starter/mood_analyzer.py#92-114)** — map score → label with `"mixed"` support:
- `score >= 2` → `"positive"`
- `score <= -2` → `"negative"`
- `score == 0` but both positive and negative words found → `"mixed"`
- `score == 0`, nothing found → `"neutral"`
- `score == 1` → `"positive"`, `score == -1` → `"negative"` (lean-thresholds)

**[explain(text)](file:///c:/Users/onika/OneDrive/Documents/Coding%20Projects/CodePath%20Projects/ai110-module3tinker-themoodmachine-starter/mood_analyzer.py#119-154)** — richer explanation:
- Identify positive hits, negative hits, and any negated tokens
- Show the final score and a readable breakdown

## Verification Plan

### Automated Tests (no formal test suite — use the labeled dataset)

Run the evaluation against `SAMPLE_POSTS` / `TRUE_LABELS`:

```
# Activate venv first
.venv\Scripts\Activate.ps1

python main.py
```

Expected: accuracy reported at end of "Rule Based Evaluation" section.
A correct implementation should get ≥ 4/6 correct on the starter dataset
(negation case "I am not happy about this" is the key test for negation handling;
"Feeling tired but kind of hopeful" tests mixed detection).

### Manual Spot-Check

After `python main.py` opens the interactive loop, test these inputs:

| Input | Expected label |
|-------|---------------|
| `I love this so much` | `positive` |
| `Today was awful` | `negative` |
| `I am not happy` | `negative` |
| `not bad at all` | `positive` |
| `this is fine` | `neutral` |
| `feeling tired but hopeful` | `mixed` |
| `soooo excited!!` | `positive` |
| `:)` | `positive` |
| `:(` | `negative` |
