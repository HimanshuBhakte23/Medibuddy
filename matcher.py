"""
matcher.py
-----------
The "TF-IDF + Cosine Similarity" block from the architecture diagram.

Builds one TF-IDF "document" per disease out of its Symptoms rows, and scores
the user's extracted keywords against every disease document using cosine
similarity — instead of the older simple "% of symptoms matched" rule.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import database


FUNCTION_WORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "with",
    "from", "by", "for", "is", "are", "after", "while", "when", "that", "it",
}


class DiseaseMatcher:
    def __init__(self, diseases):
        self.diseases = diseases
        self.corpus = [" ".join(d["symptoms"]) for d in diseases]
        # Ignore small function words ("on", "of", "in"...) so that phrases like
        # "cut on skin" / "burn on skin" are compared on their medical words
        # only. (A short custom list on purpose: scikit-learn's built-in
        # English list would also drop useful words such as "thick" or "cold".)
        self.vectorizer = TfidfVectorizer(stop_words=sorted(FUNCTION_WORDS))
        self.disease_matrix = self.vectorizer.fit_transform(self.corpus)

    def match(self, keyword_set):
        """
        keyword_set: a set/list of canonical symptom words/phrases recognized
        from the user's message (already run through the NLP pipeline +
        synonym normalization).

        Returns disease dicts sorted by cosine similarity (highest first),
        each annotated with a 0-100 'score' and the symptoms that overlapped.
        """
        if not keyword_set:
            return []

        query_text = " ".join(keyword_set)
        query_vector = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vector, self.disease_matrix)[0]

        results = []
        for disease, sim in zip(self.diseases, similarities):
            if sim <= 0:
                continue
            matched_symptoms = sorted(set(disease["symptoms"]) & set(keyword_set))
            results.append({
                **disease,
                "score": round(float(sim) * 100, 1),
                "matched_symptoms": matched_symptoms,
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results


def build_matcher():
    diseases = database.get_all_diseases_with_details()
    return DiseaseMatcher(diseases)


if __name__ == "__main__":
    database.init_db()
    matcher = build_matcher()
    tests = [
        {"fever", "mild cough", "sore throat"},
        {"itchy skin", "skin rash"},
        {"severe headache", "nausea"},
    ]
    for kws in tests:
        print(f"\nKeywords: {kws}")
        for r in matcher.match(kws)[:3]:
            print(f"  {r['name']:40s} score={r['score']:5.1f}  matched={r['matched_symptoms']}")
