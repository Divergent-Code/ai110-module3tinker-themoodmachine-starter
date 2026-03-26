"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - INTENSIFIER_WORDS: words that amplify the next sentiment word
  - SAMPLE_POSTS: short example posts for training and development
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
  - TEST_POSTS: held-out posts never seen during development
  - TEST_LABELS: human labels for each post in TEST_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    # --- expanded vocabulary ---
    "hopeful",      # fixes the "tired but hopeful" mixed-label miss
    "proud",
    "grateful",
    "wonderful",
    "fantastic",
    "glad",
    "stoked",       # slang: very excited
    "hyped",        # slang: very excited
    "blessed",
    "peaceful",
    # --- slang additions (#8 improvement) ---
    "lit",          # slang: exciting or excellent
    "fire",         # slang: very good
    "goat",         # slang: greatest of all time
    "vibing",       # slang: feeling good
    "valid",        # slang: good, acceptable
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    # --- expanded vocabulary ---
    "miserable",
    "frustrated",
    "disappointed",
    "lonely",
    "worried",
    "dread",
    "exhausted",
    "overwhelmed",
    "annoyed",
    "rough",        # colloquial: "rough day"
    # --- slang additions (#8 improvement) ---
    "mid",          # slang: mediocre or bad
    "sus",          # slang: suspicious or untrustworthy
    "trash",        # slang: terrible
    "cringe",       # slang: embarrassing or bad
    "dead",         # slang: exhausted / overwhelmed ("I'm dead")
]

# Intensifier words that double the weight of the NEXT sentiment word.
# Example: "absolutely love" -> love scores +2 instead of +1.
# Example: "so stressed" -> stressed scores -2 instead of -1.
INTENSIFIER_WORDS = [
    "so",
    "very",
    "really",
    "absolutely",
    "totally",
    "completely",
    "extremely",
    "super",
    "incredibly",
    "literally",    # colloquial intensifier
    "honestly",     # colloquial intensifier
    "lowkey",       # slang: somewhat / really (context-dependent)
    "highkey",      # slang: openly / very much
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    # --- diverse additions ---
    # Slang
    "Lowkey stressed but highkey proud of myself",
    "No cap this is the best day ever",
    # Emojis (ASCII)
    "Just got my grades back :)",
    "Missed my bus again :(",
    # Unicode emoji
    "This assignment is killing me 💀",
    # Sarcasm (hard for rule-based models — intentional edge case)
    "I absolutely love getting stuck in traffic",
    # Negation
    "Not bad at all, actually enjoyed it",
    # Mixed feelings
    "Finally done with finals but I am so exhausted",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    # --- diverse additions ---
    "mixed",     # "Lowkey stressed but highkey proud of myself"
    "positive",  # "No cap this is the best day ever" (best/ever = positive)
    "positive",  # "Just got my grades back :)"
    "negative",  # "Missed my bus again :("
    "negative",  # "This assignment is killing me 💀"
    "negative",  # "I absolutely love getting stuck in traffic" (sarcasm — model will likely predict positive; instructive miss)
    "positive",  # "Not bad at all, actually enjoyed it"
    "mixed",     # "Finally done with finals but I am so exhausted"
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)

# ---------------------------------------------------------------------
# Held-out test set (#8 improvement: real test split)
# ---------------------------------------------------------------------
# These posts are NEVER used during development or training.
# Evaluate both models here to get a fairer accuracy estimate.

TEST_POSTS = [
    "I feel so grateful for everything today",   # intensifier: so + positive
    "That exam was absolutely trash",             # slang: trash = negative
    "Not really excited but not sad either",      # negation + mixed
    "This new song is fire",                      # slang: fire = positive
    "Feeling kind of mid about the whole thing", # slang: mid = negative
    "Won the competition and I am so proud",      # intensifier: so + positive
]

TEST_LABELS = [
    "positive",  # "I feel so grateful for everything today"
    "negative",  # "That exam was absolutely trash"
    "mixed",     # "Not really excited but not sad either"
    "positive",  # "This new song is fire"
    "negative",  # "Feeling kind of mid about the whole thing"
    "positive",  # "Won the competition and I am so proud"
]
