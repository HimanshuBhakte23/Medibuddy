"""Flask file for joining files to server"""
from flask import Flask, render_template, request, jsonify
import rules
import database

app = Flask(__name__)


@app.route("/")
def home():
    """Serve the chatbot's chat interface."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat API used by static/script.js.

    Expects JSON: { "message": "I have fever and cough" }
    Returns JSON produced by rules.generate_response().
    """
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "")

    response = rules.generate_response(user_message)
    return jsonify(response)


@app.route("/health")
def health():
    """Simple route to confirm the backend + database + NLP pipeline are loaded."""
    diseases = database.get_all_diseases_with_details()
    return jsonify({
        "status": "ok",
        "database": "SQLite (medibuddy.db)",
        "conditions_loaded": len(diseases),
        "synonyms_loaded": len(rules.SYNONYM_MAP),
    })


if __name__ == "__main__":
    # debug=True is fine for a college demo; turn off for any real deployment
    app.run(debug=True, port=5000)
