"""
database.py
The architecture diagram specifies MySQL. This project uses SQLite instead,
because it needs zero server setup — the whole database is a single file
(medibuddy.db) created automatically the first time you run the app, which
makes it possible to demo on any laptop with no extra installation.
"""

import json
import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Vercel's project folder is read-only; /tmp is the only writable location.
if os.environ.get("VERCEL"):
    DB_PATH = "/tmp/medibuddy.db"
else:
    DB_PATH = os.path.join(BASE_DIR, "medibuddy.db")
DISEASE_SEED_FILE = os.path.join(BASE_DIR, "diseases_seed.json")
SYNONYM_SEED_FILE = os.path.join(BASE_DIR, "synonyms_seed.json")

SCHEMA = """
CREATE TABLE IF NOT EXISTS Disease (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    info TEXT,
    medicine_info TEXT,
    doctor TEXT
);

CREATE TABLE IF NOT EXISTS Symptoms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease_id INTEGER NOT NULL,
    symptom_text TEXT NOT NULL,
    FOREIGN KEY (disease_id) REFERENCES Disease(id)
);

CREATE TABLE IF NOT EXISTS Treatments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease_id INTEGER NOT NULL,
    precaution_text TEXT NOT NULL,
    FOREIGN KEY (disease_id) REFERENCES Disease(id)
);

CREATE TABLE IF NOT EXISTS WarningSigns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease_id INTEGER NOT NULL,
    sign_text TEXT NOT NULL,
    FOREIGN KEY (disease_id) REFERENCES Disease(id)
);

CREATE TABLE IF NOT EXISTS Synonyms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phrase TEXT NOT NULL,
    canonical_symptom TEXT NOT NULL
);
"""


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist, then sync them from the JSON seed
    files.

    The sync is additive and idempotent: any disease (matched by name) or
    synonym (matched by phrase) that is in the JSON but not yet in the
    database gets inserted; anything already there is left alone. So to add
    conditions you just edit the JSON and restart — no need to delete
    medibuddy.db any more."""
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()

    _seed_diseases(conn)
    _seed_synonyms(conn)
    conn.commit()

    conn.close()


def _seed_diseases(conn):
    with open(DISEASE_SEED_FILE, "r", encoding="utf-8") as f:
        diseases = json.load(f)

    existing = {r["name"] for r in conn.execute("SELECT name FROM Disease")}

    for d in diseases:
        if d["name"] in existing:
            continue
        cur = conn.execute(
            "INSERT INTO Disease (name, info, medicine_info, doctor) VALUES (?, ?, ?, ?)",
            (d["name"], d["info"], d["medicine_info"], d["doctor"]),
        )
        disease_id = cur.lastrowid

        for symptom in d["symptoms"]:
            conn.execute(
                "INSERT INTO Symptoms (disease_id, symptom_text) VALUES (?, ?)",
                (disease_id, symptom.lower()),
            )
        for precaution in d["precautions"]:
            conn.execute(
                "INSERT INTO Treatments (disease_id, precaution_text) VALUES (?, ?)",
                (disease_id, precaution),
            )
        for sign in d["warning_signs"]:
            conn.execute(
                "INSERT INTO WarningSigns (disease_id, sign_text) VALUES (?, ?)",
                (disease_id, sign),
            )


def _seed_synonyms(conn):
    with open(SYNONYM_SEED_FILE, "r", encoding="utf-8") as f:
        synonyms = json.load(f)
    existing = {r["phrase"] for r in conn.execute("SELECT phrase FROM Synonyms")}
    for s in synonyms:
        phrase = s["phrase"].lower()
        if phrase in existing:
            continue
        existing.add(phrase)
        conn.execute(
            "INSERT INTO Synonyms (phrase, canonical_symptom) VALUES (?, ?)",
            (s["phrase"].lower(), s["canonical_symptom"].lower()),
        )


def get_all_diseases_with_details():
    """
    Returns a list of dicts, one per disease, each with its symptoms,
    precautions and warning signs attached — this is effectively what
    a JOIN across all 4 tables gives us, reshaped for Python use.
    """
    conn = get_connection()
    diseases = conn.execute("SELECT * FROM Disease").fetchall()

    result = []
    for d in diseases:
        symptoms = [r["symptom_text"] for r in conn.execute(
            "SELECT symptom_text FROM Symptoms WHERE disease_id = ?", (d["id"],)
        )]
        precautions = [r["precaution_text"] for r in conn.execute(
            "SELECT precaution_text FROM Treatments WHERE disease_id = ?", (d["id"],)
        )]
        warning_signs = [r["sign_text"] for r in conn.execute(
            "SELECT sign_text FROM WarningSigns WHERE disease_id = ?", (d["id"],)
        )]
        result.append({
            "id": d["id"],
            "name": d["name"],
            "info": d["info"],
            "medicine_info": d["medicine_info"],
            "doctor": d["doctor"],
            "symptoms": symptoms,
            "precautions": precautions,
            "warning_signs": warning_signs,
        })
    conn.close()
    return result


def get_synonym_map():
    """Returns {phrase: canonical_symptom} straight from the Synonyms table."""
    conn = get_connection()
    rows = conn.execute("SELECT phrase, canonical_symptom FROM Synonyms").fetchall()
    conn.close()
    return {r["phrase"]: r["canonical_symptom"] for r in rows}


def get_all_symptom_vocab():
    """Every distinct symptom_text in the Symptoms table (used to recognize
    exact vocabulary even without going through TF-IDF)."""
    conn = get_connection()
    rows = conn.execute("SELECT DISTINCT symptom_text FROM Symptoms").fetchall()
    conn.close()
    return {r["symptom_text"] for r in rows}


if __name__ == "__main__":
    init_db()
    diseases = get_all_diseases_with_details()
    print(f"Database ready at {DB_PATH}")
    print(f"{len(diseases)} diseases loaded.")
    print(f"{len(get_synonym_map())} synonyms loaded.")
    print("Example row:", json.dumps(diseases[0], indent=2))
