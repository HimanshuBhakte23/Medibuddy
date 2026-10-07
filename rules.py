"""
rules.py
---------
The orchestrator. Wires together every block from the architecture diagram:

    User Input
        -> check_emergency()                    [deterministic safety check — ALWAYS first]
        -> nlp_pipeline.extract_keywords()        [Clean -> Tokenize+POS -> Lemmatize -> Keywords]
        -> intent_classifier.classify_intent()     [scikit-learn]
        -> nlp_pipeline.apply_synonyms()            [Find Synonyms, via the Synonyms table]
        -> matcher.match()                           [TF-IDF + Cosine Similarity]
        -> database.py                                [Disease / Symptoms / Treatments tables]
        -> generate_response() formats the final JSON the Flask route returns

IMPORTANT SAFETY DESIGN CHOICE:
Emergency/red-flag detection is a plain deterministic keyword check, NOT part
of the ML intent classifier. A safety-critical check should never depend on
a probabilistic model's confidence score — it either matches a known
red-flag phrase or it doesn't.
"""

import re

import database
import nlp_pipeline
import intent_classifier
import matcher as matcher_module

# ---------------------------------------------------------------------------
# Startup: build the database (first run only) and the in-memory TF-IDF matcher
# ---------------------------------------------------------------------------

database.init_db()
DISEASE_MATCHER = matcher_module.build_matcher()
SYNONYM_MAP = database.get_synonym_map()
SYMPTOM_VOCAB = database.get_all_symptom_vocab()

# ---------------------------------------------------------------------------
# Emergency / red-flag rules (unchanged: deterministic, always checked first)
# ---------------------------------------------------------------------------

EMERGENCY_RULES = {
    "Severe difficulty breathing": [
        "cant breathe", "can't breathe", "difficulty breathing", "severe breathlessness", "gasping for air", "choking",
        "saans lene me takleef", "saans phool", "saans nahi", "सांस लेने में तकलीफ", "सांस नहीं आ रही", "दम घुट रहा"
    ],
    "Loss of consciousness": [
        "unconscious", "fainted", "passed out", "not waking up", "unresponsive",
        "behosh", "behoshi", "chakkar aakar gir", "बेहोश", "बेहोशी"
    ],
    "Severe bleeding": [
        "heavy bleeding", "severe bleeding", "bleeding a lot", "won't stop bleeding", "blood loss",
        "khoon nikal raha", "bohot khoon", "खून बह रहा", "खून निकल रहा"
    ],
    "Seizure": [
        "seizure", "convulsion", "fit", "fits", "shaking uncontrollably",
        "daura", "mirgi", "दौरा", "मिर्गी"
    ],
    "Sudden weakness / possible stroke": [
        "sudden weakness", "face drooping", "slurred speech", "one side numb", "cannot move one side",
        "chehra tedha", "lakwa", "लकवा"
    ],
    "Severe chest pain": [
        "severe chest pain", "chest pain radiating", "crushing chest pain", "chest pain and sweating",
        "seene me dard", "seene me tez dard", "chaati me dard", "सीने में दर्द", "छाती में दर्द", "सीने में तेज दर्द"
    ],
    "Snake bite": [
        "snake bite", "snakebite", "bitten by a snake", "bitten by snake", "snake bit",
        "snake ne kat", "saanp ne kat", "sanp ne kat", "saap ne kat", "saanp kat", "saanp ka kat",
        "सांप ने काट", "साँप ने काट", "सांप के काट", "साँप के काट", "सांप काट"
    ],
    "Coughing or vomiting blood": [
        "coughing up blood", "cough with blood", "coughing blood", "vomiting blood", "vomiting with blood",
        "vomit blood", "blood in vomit",
        "khoon ki ulti", "khoon ki qai", "खून की उल्टी", "खून की खांसी"
    ],
    "Severe burn or electric shock": [
        "severe burn", "severe burns", "electric shock", "electrocuted", "chemical burn", "burnt badly",
        "burned badly", "burns all over", "bijli ka jhatka", "karant lag", "करंट लग", "बिजली का झटका"
    ],
    "Severe injury (bone visible / head injury with vomiting)": [
        "bone sticking out", "bone is visible", "bone visible", "open fracture", "bone coming out",
        "hadi bahar", "हड्डी बाहर", "head injury and vomiting", "hit my head and vomiting"
    ],
}

# Short words that would otherwise match inside unrelated words (e.g. "fit"
# inside "fitness"/"benefit"). These must match as a whole word; every other
# phrase keeps the original simple substring match (so Hindi stems like
# "saans phool" still match "saans phoolna").
WHOLE_WORD_PHRASES = {"fit", "fits", "severe burn", "severe burns"}


def _phrase_in_text(phrase, text):
    if phrase in WHOLE_WORD_PHRASES:
        return re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", text) is not None
    return phrase in text


def check_emergency(user_text):
    text = user_text.lower()
    for label, phrases in EMERGENCY_RULES.items():
        for phrase in phrases:
            if _phrase_in_text(phrase, text):
                return label
    return None


# ---------------------------------------------------------------------------
# Small-talk responses (for greeting / thanks / goodbye intents)
# ---------------------------------------------------------------------------

SMALLTALK_REPLIES = {
    "greeting": "Hi there! I'm MediBuddy. Tell me what symptoms you're feeling and I'll try to help.",
    "thanks": "You're welcome! Take care, and don't hesitate to check in again if you feel unwell.",
    "goodbye": "Take care! Remember to see a doctor if your symptoms continue or get worse.",
}

# Only trust the ML classifier for small talk when it's both confident AND
# the message is short — a long symptom sentence shouldn't get hijacked by a
# stray high-probability guess.
INTENT_CONFIDENCE_THRESHOLD = 0.30
MAX_WORDS_FOR_SMALLTALK = 6

MIN_SIMILARITY_SCORE = 5.0          # below this, treat as "no real match"
LOW_CONFIDENCE_SCORE_THRESHOLD = 34  # below this, ask for more details too


def generate_response(user_text):
    """Main entry point used by the Flask backend (app.py)."""
    if not user_text or not user_text.strip():
        return {
            "type": "clarify",
            "message": "Please tell me the symptoms you're experiencing (for example: \"I have a headache and sore throat\")."
        }

    # --- Step 1: Emergency red-flag check (ALWAYS runs first, deterministic) ---
    emergency = check_emergency(user_text)
    if emergency:
        return {
            "type": "emergency",
            "matched": emergency,
            "message": (
                f"⚠️ This sounds like it could be a medical emergency ({emergency}). "
                "Please seek immediate/emergency medical care or contact your local "
                "emergency number right now. This chatbot cannot handle emergencies."
            )
        }

    # --- Step 2: NLP pipeline — clean, tokenize+POS, lemmatize, extract keywords ---
    cleaned_text, keywords = nlp_pipeline.extract_keywords(user_text)

    # --- Step 3: Intent detection (scikit-learn) ---
    intent, confidence = intent_classifier.classify_intent(cleaned_text)
    word_count = len(cleaned_text.split())

    if (
        intent in SMALLTALK_REPLIES
        and confidence >= INTENT_CONFIDENCE_THRESHOLD
        and word_count <= MAX_WORDS_FOR_SMALLTALK
    ):
        return {
            "type": "smalltalk",
            "detected_intent": intent,
            "message": SMALLTALK_REPLIES[intent],
        }

    # --- Step 4: Synonym matching against the Synonyms table ---
    found_symptoms = nlp_pipeline.apply_synonyms(cleaned_text, keywords, SYNONYM_MAP, SYMPTOM_VOCAB)

    if not found_symptoms:
        return {
            "type": "clarify",
            "detected_intent": intent,
            "extracted_keywords": keywords,
            "message": (
                "I couldn't recognize any specific symptoms in that message. "
                "Could you describe what you're feeling in more detail? "
                "For example: fever, cough, sore throat, stomach pain, etc."
            )
        }

    # --- Step 5: TF-IDF + Cosine Similarity matching against the SQL database ---
    matches = DISEASE_MATCHER.match(found_symptoms)
    matches = [m for m in matches if m["score"] >= MIN_SIMILARITY_SCORE]

    if not matches:
        return {
            "type": "clarify",
            "detected_intent": intent,
            "found_symptoms": sorted(found_symptoms),
            "message": (
                "I noted these symptoms: " + ", ".join(sorted(found_symptoms)) + ". "
                "That's not enough to match a possible condition yet — could you share "
                "any other symptoms you're noticing?"
            )
        }

    top = matches[:3]
    needs_more_info = (
        len(found_symptoms) <= 1
        or top[0]["score"] < LOW_CONFIDENCE_SCORE_THRESHOLD
    )

    return {
        "type": "result",
        "detected_intent": intent,
        "extracted_keywords": keywords,
        "found_symptoms": sorted(found_symptoms),
        "matches": top,
        "needs_more_info": needs_more_info,
        "disclaimer": (
            "These are possible related conditions based on NLP keyword extraction and "
            "TF-IDF similarity matching against our database, NOT a confirmed diagnosis. "
            "The match score is a text-similarity percentage, not a medical probability. "
            "Please consult a qualified doctor for an accurate diagnosis."
        )
    }


if __name__ == "__main__":
    import json
    test_inputs = [
        "hi there",
        "I have fever, cough and sore throat",
        "my skin is itchy and I have a rash",
        "I can't breathe properly and my chest hurts a lot",
        "I feel a bit off",
        "thanks a lot",
        "",
    ]
    for t in test_inputs:
        print("\nUSER:", t)
        print(json.dumps(generate_response(t), indent=2))
