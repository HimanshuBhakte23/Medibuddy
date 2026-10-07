"""
intent_classifier.py
----------------------
The "Intent Detection (scikit-learn)" block from the architecture diagram.

A small TF-IDF + Logistic Regression text classifier trained on a hand-built
example dataset, distinguishing:

    greeting        -> "hi", "hello", "good morning"
    thanks          -> "thank you", "thanks a lot"
    goodbye         -> "bye", "see you"
    symptom_query   -> "I have fever and cough", "my head hurts"
    other           -> off-topic small talk ("who are you", "tell me a joke")

NOTE: this is deliberately a *small* demo classifier — it exists to show the
ML "Intent Detection" stage from the architecture diagram. Emergency/red-flag
detection is intentionally NOT handled here — that stays a separate,
deterministic keyword check in rules.py, because a safety-critical check
should never depend on a probabilistic model's confidence score.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

TRAINING_DATA = {
    "greeting": [
        "hi", "hello", "hey", "hey there", "good morning", "good evening",
        "hiya", "hello there", "morning", "hi there",
        "namaste", "namaskar", "pranam", "नमस्ते", "नमस्कार",
    ],
    "thanks": [
        "thank you", "thanks a lot", "thanks", "thank you so much",
        "appreciate it", "thanks for the help", "much appreciated",
        "dhanyawad", "shukriya", "bahut shukriya", "धन्यवाद", "शुक्रिया",
    ],
    "goodbye": [
        "bye", "goodbye", "see you", "talk later", "see you later",
        "catch you later", "bye bye", "alvida", "अलविदा",
    ],
    "symptom_query": [
        "i have fever and cough", "my head hurts a lot",
        "i feel nauseous and dizzy", "i have a rash on my arm",
        "my stomach is paining since morning", "i am having chest congestion and cold",
        "feeling dizzy and weak today", "i have sore throat and body ache",
        "my eyes are red and itchy", "i have been vomiting since last night",
        "there is burning sensation while urinating", "i have a runny nose and sneezing",
        "my throat hurts when i swallow", "i feel tired and have no appetite",
        "i have severe headache with nausea", "my skin is itchy and there is a rash",
        "i have diarrhea and stomach cramps", "i feel feverish and my body aches",
        "mujhe bukhar hai", "sar me dard hai", "pet dard ho raha hai",
        "mujhe ulti aur dast hai", "mujhe bukhar aur khansi hai",
        "मुझे बुखार है", "सिर में दर्द है", "पेट दर्द हो रहा है",
        "मुझे उल्टी आ रही है", "बुखार और खांसी है",
    ],
    "other": [
        "who are you", "what can you do", "tell me a joke",
        "what is the weather today", "what is your name",
        "how does this chatbot work", "are you a real doctor",
        "aap kaun ho", "kya kar sakte ho", "आप कौन हो",
    ],
}


def _build_training_rows():
    texts, labels = [], []
    for label, examples in TRAINING_DATA.items():
        for ex in examples:
            texts.append(ex)
            labels.append(label)
    return texts, labels


def build_intent_pipeline():
    """Trains (in-memory, on startup) a tiny TF-IDF + Logistic Regression
    intent classifier. Training on ~40 short examples takes a fraction of
    a second, so there's no need to persist the model to disk."""
    texts, labels = _build_training_rows()
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(texts, labels)
    return pipeline


_INTENT_PIPELINE = build_intent_pipeline()


def classify_intent(cleaned_text):
    """
    Returns (label, confidence) for a cleaned (lowercased, punctuation-free)
    piece of text. Confidence is the model's own predicted probability for
    the winning class.
    """
    if not cleaned_text.strip():
        return "other", 0.0

    proba = _INTENT_PIPELINE.predict_proba([cleaned_text])[0]
    classes = _INTENT_PIPELINE.classes_
    best_idx = proba.argmax()
    return classes[best_idx], float(proba[best_idx])


if __name__ == "__main__":
    tests = [
        "hi there", "thanks a lot", "bye for now",
        "i have fever and sore throat", "who made you",
        "my stomach hurts and i feel like vomiting",
    ]
    for t in tests:
        label, conf = classify_intent(t)
        print(f"{t!r:45s} -> {label:15s} ({conf:.2f})")
