# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
import string
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS, INTENSIFIER_WORDS

# ---------------------------------------------------------------------------
# Emoji lookup tables
# ---------------------------------------------------------------------------

# ASCII emoji strings mapped to a sentiment sentinel token.
_ASCII_EMOJI_MAP: Dict[str, str] = {
    ":)": "__emoji_positive__",
    ":-)":  "__emoji_positive__",
    "=)": "__emoji_positive__",
    ":D": "__emoji_positive__",
    ":-D": "__emoji_positive__",
    "=D": "__emoji_positive__",
    ":(": "__emoji_negative__",
    ":-(": "__emoji_negative__",
    "=(": "__emoji_negative__",
    ":/": "__emoji_negative__",
    ">:(": "__emoji_negative__",
    ">:|": "__emoji_negative__",
}

# Unicode codepoint ranges that are commonly used emoji.
# Rather than enumerate every emoji, we check the Unicode category.
def _is_unicode_emoji(char: str) -> bool:
    """Return True if the character is a Unicode emoji."""
    cp = ord(char)
    return (
        0x1F600 <= cp <= 0x1F64F  # emoticons
        or 0x1F300 <= cp <= 0x1F5FF  # misc symbols & pictographs
        or 0x1F680 <= cp <= 0x1F6FF  # transport & map
        or 0x1F900 <= cp <= 0x1F9FF  # supplemental symbols
        or 0x2600  <= cp <= 0x26FF   # misc symbols
        or 0x2700  <= cp <= 0x27BF   # dingbats
    )


# Negator words that flip the sentiment of the next content word.
_NEGATORS = {"not", "never", "no", "cant", "can't", "dont", "don't", "wont", "won't"}



class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
        intensifier_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS
        intensifier_words = intensifier_words if intensifier_words is not None else INTENSIFIER_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)
        self.intensifier_words = set(w.lower() for w in intensifier_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Improvements applied:
          - Replaces ASCII emojis (":)", ":-(") with sentiment sentinel tokens
            before splitting so they aren't swallowed by punctuation removal.
          - Replaces Unicode emojis (😂, 🥲, 💀) with sentiment sentinel tokens.
          - Removes remaining punctuation from each token.
          - Normalizes repeated characters ("soooo" -> "soo").

        Args:
            text: Raw input string from the user.

        Returns:
            A list of lowercase string tokens ready for scoring.
        """
        # --- Step 1: replace ASCII emojis BEFORE lowercasing/splitting so
        #   that case-sensitive patterns like ":D" are caught first.
        for emoji_str, sentinel in _ASCII_EMOJI_MAP.items():
            text = text.replace(emoji_str, f" {sentinel} ")

        # --- Step 2: lowercase and handle Unicode emojis character-by-character.
        cleaned_chars: List[str] = []
        for char in text.lower():
            if _is_unicode_emoji(char):
                # We can't easily tell positive vs negative from a single
                # Unicode codepoint without a full lookup table, so we add
                # the raw character as its own token for now.  Subclasses or
                # future iterations can map these to sentinels like we do for
                # ASCII emojis.
                cleaned_chars.append(f" {char} ")
            else:
                cleaned_chars.append(char)
        cleaned = "".join(cleaned_chars).strip()

        # --- Step 3: split into raw tokens.
        raw_tokens = cleaned.split()

        tokens: List[str] = []
        for token in raw_tokens:
            # Keep sentinel tokens produced by the emoji step untouched.
            if token.startswith("__") and token.endswith("__"):
                tokens.append(token)
                continue

            # --- Step 4: strip surrounding punctuation from the token so
            #   words like "great!" or "'happy'" still match the lexicon.
            token = token.strip(string.punctuation)

            if not token:
                continue

            # --- Step 5: normalize runs of 3+ identical letters to 2.
            #   "soooo" -> "soo", "terribleee" -> "terribleee"->"terriblee"
            token = re.sub(r"(.)\1{2,}", r"\1\1", token)

            tokens.append(token)

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        The improvements implemented here are:
          - Negation handling: "not happy" flips the score of "happy" from +1 to -1.
            Negator words are: not, never, no, can't, don't, won't.
          - Extended negation window (#8 improvement): the negation flag survives
            one non-sentiment token, so "not at all happy" still negates "happy".
          - Intensifiers (#8 improvement): words like "so", "very", "absolutely"
            double the score of the immediately following sentiment word.
          - Emoji signals: ASCII/Unicode emoji sentinels count double (+2 / -2)
            so they have a stronger influence than a single content word.

        Args:
            text: Raw input string from the user.

        Returns:
            An integer sentiment score (positive = good mood, negative = bad mood).
        """
        tokens = self.preprocess(text)
        score = 0
        negated = False         # True when we are inside a negation window.
        negation_budget = 0     # How many non-sentiment tokens the window can skip.
        intensified = False     # True when the previous token was an intensifier.

        for token in tokens:
            # --- Emoji sentinels: strong positive or negative signal.
            if token == "__emoji_positive__":
                weight = 2
                if intensified:
                    weight *= 2
                score += -weight if negated else weight
                negated = False
                negation_budget = 0
                intensified = False
                continue
            if token == "__emoji_negative__":
                weight = 2
                if intensified:
                    weight *= 2
                score += weight if negated else -weight
                negated = False
                negation_budget = 0
                intensified = False
                continue

            # --- Negator words: open a 2-word negation window.
            if token in _NEGATORS:
                negated = True
                negation_budget = 2  # can skip up to 2 non-sentiment tokens
                intensified = False
                continue

            # --- Intensifier words: set flag, don't score.
            if token in self.intensifier_words:
                intensified = True
                # Intensifiers don't consume the negation window.
                continue

            # --- Sentiment words: apply negation + intensification.
            if token in self.positive_words:
                weight = 2 if intensified else 1
                score += -weight if negated else weight
                negated = False
                negation_budget = 0
                intensified = False
            elif token in self.negative_words:
                weight = 2 if intensified else 1
                score += weight if negated else -weight
                negated = False
                negation_budget = 0
                intensified = False
            else:
                # Non-sentiment, non-negator, non-intensifier token.
                # Consume one slot of the negation window if active.
                if negated and negation_budget > 0:
                    negation_budget -= 1
                    if negation_budget == 0:
                        negated = False  # window exhausted

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The mapping used here is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0, but both positive and negative words present -> "mixed"
          - score == 0, nothing found -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use higher thresholds (e.g. score >= 2 for "positive")
          - Expand the "mixed" detection to cover scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.

        Args:
            text: Raw input string from the user.

        Returns:
            One of: "positive", "negative", "neutral", or "mixed".
        """
        score = self.score_text(text)

        if score > 0:
            return "positive"
        if score < 0:
            return "negative"

        # Score is exactly 0 — check whether the tie is because both
        # positive and negative words are present (mixed) or because
        # nothing was found (neutral).
        tokens = self.preprocess(text)
        has_positive = any(t in self.positive_words for t in tokens)
        has_negative = any(t in self.negative_words for t in tokens)

        if has_positive and has_negative:
            return "mixed"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        This implementation tracks negation so that a word like "happy"
        after "not" is shown in the negated list rather than the positive list.

        Example output:
          'Score = -1 | positive: [] | negative: ["happy" (negated)] | label: negative'

        Args:
            text: Raw input string from the user.

        Returns:
            A human-readable string summarising the model's reasoning.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []   # words that added to the score
        negative_hits: List[str] = []   # words that subtracted from the score
        negated_hits: List[str] = []    # sentiment words whose sign was flipped
        score = 0
        negated = False

        for token in tokens:
            if token == "__emoji_positive__":
                if negated:
                    negated_hits.append(":) (negated)")
                    score -= 2
                else:
                    positive_hits.append(":)")
                    score += 2
                negated = False
                continue
            if token == "__emoji_negative__":
                if negated:
                    negated_hits.append(":( (negated)")
                    score += 2
                else:
                    negative_hits.append(":(")
                    score -= 2
                negated = False
                continue
            if token in _NEGATORS:
                negated = True
                continue
            if token in self.positive_words:
                if negated:
                    negated_hits.append(f"{token} (negated)")
                    score -= 1
                else:
                    positive_hits.append(token)
                    score += 1
                negated = False
            elif token in self.negative_words:
                if negated:
                    negated_hits.append(f"{token} (negated)")
                    score += 1
                else:
                    negative_hits.append(token)
                    score -= 1
                negated = False

        label = self.predict_label(text)
        parts = [
            f"Score = {score}",
            f"positive: {positive_hits or []}",
            f"negative: {negative_hits or []}",
        ]
        if negated_hits:
            parts.append(f"negated: {negated_hits}")
        parts.append(f"label: {label}")
        return " | ".join(parts)
