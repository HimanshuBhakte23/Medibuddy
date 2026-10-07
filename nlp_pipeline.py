"""
nlp_pipeline.py
----------------
The "Dynamic NLP Processing" block from the architecture diagram:

    Clean Text -> Tokenize (Tokenization + POS) -> Lemmatize -> Extract Keywords
                                                                       |
                                                                 Find Synonyms

Implemented with NLTK (the diagram lists NLTK for text cleaning and spaCy for
keyword extraction — this project uses NLTK for the whole pipeline so there's
a single, lighter dependency to install; the pipeline stages are the same).
"""

import re
import os
import nltk

# Make sure NLTK can find data installed in the user's AppData folder
nltk_data_path = os.path.join(
    os.path.expanduser("~"),
    "AppData",
    "Roaming",
    "nltk_data"
)

if os.path.exists(nltk_data_path):
    nltk.data.path.append(nltk_data_path)

# ---------------------------------------------------------------------------
# One-time NLTK data setup
# ---------------------------------------------------------------------------
# These downloads only actually happen once per machine (NLTK caches them
# under ~/nltk_data). An internet connection is needed the first time only.


_REQUIRED_NLTK_PACKAGES = [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/stopwords", "stopwords"),
]


def ensure_nltk_data():
    missing = []

    for find_path, package_name in _REQUIRED_NLTK_PACKAGES:
        try:
            nltk.data.find(find_path)
        except LookupError:
            # WordNet can also be installed as a ZIP archive.
            if package_name == "wordnet":
                try:
                    nltk.data.find("corpora/wordnet.zip")
                    continue
                except LookupError:
                    pass

            missing.append(package_name)

    if missing:
        raise LookupError(
            "Missing NLTK packages: " + ", ".join(missing)
            + ". Install them before starting the app."
        )


ensure_nltk_data()

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english")) - {"no", "not"}  # keep negation words
LEMMATIZER = WordNetLemmatizer()

# Keep only these POS categories as candidate "keywords" (symptom-bearing words)
KEEP_POS_PREFIXES = ("NN", "JJ", "VB")


# ---------------------------------------------------------------------------
# Stage 1: Clean Text
# ---------------------------------------------------------------------------

def clean_text(text):
    """Lowercase + strip punctuation/extra whitespace, preserving English and Devanagari script."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\u0900-\u097F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Stage 2: Tokenize (Tokenization + POS)
# ---------------------------------------------------------------------------

def tokenize_and_tag(cleaned_text):
    tokens = word_tokenize(cleaned_text)
    return nltk.pos_tag(tokens)  # [(word, POS), ...]


# ---------------------------------------------------------------------------
# Stage 3: Lemmatize
# ---------------------------------------------------------------------------

def _wordnet_pos(pos_tag):
    if pos_tag.startswith("J"):
        return wordnet.ADJ
    if pos_tag.startswith("V"):
        return wordnet.VERB
    if pos_tag.startswith("N"):
        return wordnet.NOUN
    if pos_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_tagged(tagged_tokens):
    return [LEMMATIZER.lemmatize(word, _wordnet_pos(pos)) for word, pos in tagged_tokens]


# ---------------------------------------------------------------------------
# Stage 4: Extract Keywords
# ---------------------------------------------------------------------------

def extract_keywords(text):
    """
    Runs the full pipeline: Clean -> Tokenize+POS -> Lemmatize -> Extract Keywords.

    Returns:
        cleaned_text (str): the cleaned full sentence (still useful for
                             phrase-based synonym lookups)
        keywords (list[str]): lemmatized nouns/adjectives/verbs, stopwords
                               removed — these represent the "medically
                               meaningful" words in the sentence
    """
    cleaned = clean_text(text)
    if not cleaned:
        return cleaned, []

    tagged = tokenize_and_tag(cleaned)
    lemmas = lemmatize_tagged(tagged)

    keywords = []
    for (word, pos), lemma in zip(tagged, lemmas):
        if word in STOPWORDS:
            continue
        is_devanagari = any("\u0900" <= char <= "\u097F" for char in word)
        if pos.startswith(KEEP_POS_PREFIXES) or is_devanagari:
            keywords.append(lemma)

    return cleaned, keywords


# ---------------------------------------------------------------------------
# Synonym normalization (uses the Synonyms table from database.py)
# ---------------------------------------------------------------------------

def apply_synonyms(cleaned_text, keywords, synonym_map, symptom_vocab):
    """
    Combines:
      - phrase-level synonym matches found directly in the cleaned sentence
        (handles multi-word synonyms like "throwing up" -> "vomiting")
      - keyword-level matches against the known symptom vocabulary, including
        a word-overlap fallback for multi-word canonical symptoms so that
        phrasing like "my skin is itchy and I have a rash" still links to
        "itchy skin" / "skin rash" even when the words aren't adjacent.

    Returns a set of canonical symptom strings recognized from the input.
    """
    found = set()
    tokens = set(keywords) | set(cleaned_text.split())

    # 1) phrase-based synonyms
    for phrase, canonical in sorted(synonym_map.items(), key=lambda x: -len(x[0])):
        if phrase in cleaned_text:
            found.add(canonical)

    # 2) direct + word-overlap match against the real symptom vocabulary
    for symptom in sorted(symptom_vocab, key=lambda s: -len(s)):
        if symptom in cleaned_text:
            found.add(symptom)
            continue
        words = symptom.split()
        if len(words) > 1 and all(w in tokens for w in words):
            found.add(symptom)

    return found


if __name__ == "__main__":
    samples = [
        "I have fever, cough and sore throat",
        "my skin is itchy and I have a rash",
        "I'm feeling really tired and my head is throbbing",
    ]
    for s in samples:
        cleaned, kws = extract_keywords(s)
        print(f"\nInput: {s}")
        print("Cleaned:", cleaned)
        print("Keywords:", kws)
