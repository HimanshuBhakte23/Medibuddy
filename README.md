# MediBuddy — Dynamic NLP-Based Medical Chatbot

A symptom-guidance chatbot that runs real NLP text processing, an ML intent
classifier, TF-IDF + cosine similarity matching, and a SQL database — matching
the "Dynamic NLP-Based Chatbot" system architecture diagram.

---

## 1. Architecture (matches the diagram)

```
User Input
   │
   ▼
Frontend (HTML, CSS, JS)
   │
   ▼
Flask Backend  (app.py)
   │
   ▼
Emergency check (deterministic, ALWAYS first — see note below)
   │
   ▼
Dynamic NLP Processing   (nlp_pipeline.py, using NLTK)
   Clean Text → Tokenize (Tokenization + POS) → Lemmatize → Extract Keywords
   │
   ▼
Intent Detection  (intent_classifier.py, scikit-learn: TF-IDF + Logistic Regression)
   │
   ▼
Synonym Matching  (Synonyms table, looked up via database.py)
   │
   ▼
TF-IDF + Cosine Similarity  (matcher.py, scikit-learn)
   │
   ▼
SQL Database  (database.py — SQLite; tables: Disease, Symptoms, Treatments, Synonyms, WarningSigns)
   │
   ▼
Best Matching Information found in SQL
   │
   ▼
Python formats the JSON response  (rules.py)
   │
   ▼
Output displayed in the chat UI  (static/script.js)
```

### One deliberate deviation from the diagram — and why

**MySQL → SQLite.** The diagram specifies MySQL. This project uses SQLite
instead, because it needs *zero server setup* — the entire database is a
single file (`medibuddy.db`) that's created and filled in automatically the
first time you run `app.py`. That means you can demo this on any laptop
with nothing extra installed. The SQL and the table design (`Disease`,
`Symptoms`, `Treatments`, `Synonyms`, plus `WarningSigns`) are the same
either way — to point this at a real MySQL server later, you would only:

1. `pip install mysql-connector-python`
2. Swap the `sqlite3.connect(...)` line in `database.py` for a MySQL connection
3. Change `AUTOINCREMENT` → `AUTO_INCREMENT` in the `CREATE TABLE` statements

Nothing else (`rules.py`, `matcher.py`, `app.py`, the frontend) would need to change.

**Keyword Extraction: spaCy → NLTK.** The diagram lists spaCy for keyword
extraction and NLTK for text cleaning. This project uses NLTK for the whole
NLP pipeline (cleaning, tokenizing, POS tagging, lemmatizing, and keyword
extraction) so there's a single, lighter dependency to install instead of
two separate NLP libraries. The pipeline stages are identical either way.

**Emergency detection stays rule-based, not ML.** Red-flag phrases (severe
chest pain, can't breathe, loss of consciousness, etc.) are checked with a
plain deterministic keyword match *before* anything else runs — not through
the scikit-learn intent classifier. A safety-critical check should never
depend on a probabilistic model's confidence score.

---

## 2. Project structure

```
Medical_Chatbot/
├── app.py                  Flask routes: / and /chat
├── rules.py                 Orchestrator — wires every stage together
├── nlp_pipeline.py            NLTK: clean → tokenize+POS → lemmatize → keywords
├── intent_classifier.py         scikit-learn intent detection (greeting/thanks/goodbye/symptom_query/other)
├── matcher.py                     scikit-learn TF-IDF + cosine similarity
├── database.py                      SQLite schema, seeding, and queries
├── diseases_seed.json                 Seed data loaded into the DB on first run
├── synonyms_seed.json                  Seed data for the Synonyms table
├── requirements.txt
├── templates/index.html
└── static/style.css + script.js
```

`medibuddy.db` (the SQLite database file) is included, pre-built with all 43
conditions. If it is ever missing or out of date, the app rebuilds / updates
it automatically from the two JSON seed files on startup.

## 3. How to run it

```bash
cd Medical_Chatbot
pip install -r requirements.txt
python app.py
```

The **first run only** also needs internet access once, to download a few
small NLTK data packages (tokenizer, POS tagger, lemmatizer, stopwords —
a few MB total). NLTK caches these on your machine, so every run after
that works fully offline.

Then open **http://127.0.0.1:5000**.

## 4. What's new in this NLP version vs. a plain rule-based bot

- **Visible pipeline trace in the UI.** Every symptom-matching reply shows
  the detected intent and the exact keywords the NLP pipeline extracted
  (e.g. `Intent: symptom_query` · `Keywords: fever, cough, sore, throat`) —
  useful to point at during your viva to prove the NLP stages are really running.
- **TF-IDF match instead of a fixed rule score.** Matching now uses cosine
  similarity between the user's extracted keywords and each disease's
  symptom text in the database, so partial/fuzzy overlaps are scored
  more smoothly than plain "X out of Y symptoms matched."
- **Real intent detection.** Short greetings ("hi", "thanks", "bye") get a
  natural small-talk reply instead of being forced through symptom matching.
- **A real (if small) SQL database** replacing the flat JSON file, with
  normalized tables you can inspect directly:
  ```bash
  python3 -c "import sqlite3; c=sqlite3.connect('medibuddy.db'); print(c.execute('SELECT name FROM Disease').fetchall())"
  ```

## 5. Quick test cases to show your teacher

| Input | What it demonstrates |
|---|---|
| "hi" | ML intent detection → greeting small talk |
| "I have fever, cough and sore throat" | Full pipeline: NLP keywords → TF-IDF ranking across 3 conditions |
| "my skin is itchy and I have a rash" | Word-order-independent keyword + synonym matching |
| "I have severe chest pain" | Deterministic emergency check bypasses NLP/ML entirely |
| "I feel a bit off" | Clarify flow when no medically-meaningful keyword is extracted |
| "thanks a lot" | Intent detection → thanks small talk |
| "cough for 3 weeks, night sweats, weight loss and fever" | Multi-symptom match → Tuberculosis |
| "I burnt my hand with hot oil and it has blisters" | Wound matching → Minor Burn |
| "kutte ne kata" | Hindi/Hinglish synonym → Dog / Animal Bite |
| "saanp ne kata" | Deterministic emergency rule → snake bite alert |

## 6. Extending it

- Add more conditions: append to `diseases_seed.json` (and any new phrases to
  `synonyms_seed.json`) and restart. Startup now syncs the database
  additively — new diseases/synonyms are inserted, existing ones are left
  alone — so you no longer need to delete `medibuddy.db`.
  Each entry needs: `name`, `symptoms`, `info`, `medicine_info`, `doctor`,
  `precautions`, `warning_signs`. Tip: reuse existing symptom names where you
  can, and make sure every `canonical_symptom` in the synonyms file is
  actually listed under at least one disease.
- Add more synonyms: same idea, in `synonyms_seed.json`.
- Add more intent examples: edit `TRAINING_DATA` in `intent_classifier.py`.
- Swap to real MySQL: see the note in Section 1.

---

**Medical safety note:** This tool never claims a confirmed diagnosis and
never recommends a specific prescription/dosage. Emergency symptoms are
always routed to "seek professional care immediately," independent of the
NLP/ML pipeline. Keep this framing in your presentation and viva answers.

---

## 7. Conditions covered (43)

**Original 13:** Common Cold, Influenza (Flu), Allergic Rhinitis, Migraine, Gastritis, Food Poisoning, Conjunctivitis (Pink Eye), Skin Allergy (Contact Dermatitis / Urticaria), Sinusitis, Sore Throat (Pharyngitis), Urinary Tract Infection (UTI), Tension Headache, Dehydration.

**Added — 19 common diseases:** Type 2 Diabetes, Hypertension (High Blood Pressure), Asthma, Pneumonia, Acute Bronchitis, Tuberculosis (TB), Dengue Fever, Malaria, Typhoid Fever, COVID-19, Jaundice (Hepatitis), Acid Reflux (GERD), Anemia, Chickenpox, Kidney Stones, Appendicitis, Ear Infection, Ringworm (Fungal Skin Infection), Scabies.

**Added — 11 common wounds / injuries:** Minor Cut (Laceration), Scrape (Abrasion), Minor Burn, Sprain, Suspected Fracture (Broken Bone), Dog / Animal Bite, Snake Bite, Insect Bite or Sting, Nosebleed, Bruise (Contusion), Puncture Wound (Nail / Thorn / Needle).

Synonyms grew from 126 to 1,041 phrases (English, Hinglish and Hindi), e.g.
"kutte ne kata", "moch aa gayi", "naak se khoon", "can't smell".

### What changed in this version (besides the data)
- `database.py` — startup sync is now additive/idempotent (see Section 6).
- `matcher.py` — TF-IDF now ignores tiny function words ("on", "of", "in"…) so
  phrases like "cut on skin" aren't ranked by filler words.
- `rules.py` — new always-first emergency rules: snake bite, coughing/vomiting
  blood, severe burn / electric shock, bone visible. Also fixed a bug where the
  seizure keyword "fit" matched inside words like "fitness" or "benefit".
- Wound entries (cuts, burns, bites, fractures…) follow the same safety framing
  as the rest of the bot: general first-aid and "seek care if…" signs, no
  dosages, no diagnosis.
