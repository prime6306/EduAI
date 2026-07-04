"""
tests/test_backend.py – Unit tests for EduAI backend modules.
"""
import pytest, os, sys

# Patch env before imports
os.environ.setdefault("GROQ_API_KEY", "test_key")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017/eduai_test")
os.environ.setdefault("JWT_SECRET", "test_secret")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Auth tests ─────────────────────────────────────────────
def test_password_hash_and_verify():
    from backend.auth.utils import hash_password, verify_password
    hashed = hash_password("mypassword")
    assert verify_password("mypassword", hashed)
    assert not verify_password("wrongpassword", hashed)


def test_create_decode_token():
    from backend.auth.utils import create_access_token, decode_token
    token = create_access_token({"sub": "test@example.com"})
    payload = decode_token(token)
    assert payload is not None
    assert payload["sub"] == "test@example.com"


def test_invalid_token():
    from backend.auth.utils import decode_token
    result = decode_token("invalid.token.here")
    assert result is None


# ── NLP tests ─────────────────────────────────────────────
def test_preprocess_text():
    from backend.modules.nlp.pipeline import _preprocess
    result = _preprocess("The quick brown fox jumps over the lazy dog!")
    assert isinstance(result, str)
    assert "the" not in result.lower().split()  # stopword removed


def test_ngram_similarity_identical():
    from backend.modules.nlp.pipeline import ngram_similarity
    text = "This is a test document about electronics and signal processing."
    score = ngram_similarity(text, text)
    assert score == pytest.approx(1.0, abs=0.01)


def test_ngram_similarity_different():
    from backend.modules.nlp.pipeline import ngram_similarity
    t1 = "The Fourier transform is used in signal processing"
    t2 = "Machine learning algorithms include neural networks"
    score = ngram_similarity(t1, t2)
    assert score < 0.5


def test_plagiarism_detection_flags_similar():
    from backend.modules.nlp.pipeline import detect_plagiarism
    texts = {
        "Alice": "The Fourier transform decomposes a function into its constituent frequencies and is widely used in signal processing.",
        "Bob":   "The Fourier transform decomposes functions into constituent frequencies and is used widely in signal processing.",
        "Carol": "Machine learning is a subset of artificial intelligence focused on learning from data.",
    }
    results = detect_plagiarism(texts, threshold=0.5)
    flagged_pairs = [(r["student1"], r["student2"]) for r in results]
    assert ("Alice", "Bob") in flagged_pairs or ("Bob", "Alice") in flagged_pairs


# ── Dropout model tests ───────────────────────────────────
def test_dropout_predict_returns_valid_keys():
    from backend.modules.dropout.model import predict_dropout
    features = {
        "age": 18, "studytime": 2, "failures": 1, "absences": 10,
        "G1": 8, "G2": 7, "sex": "M", "address": "U",
        "schoolsup": "no", "famsup": "yes",
        "freetime": 3, "goout": 4, "health": 3, "famrel": 3,
    }
    # This trains a model if not present
    result = predict_dropout(features, model_type="rf")
    assert "prediction" in result
    assert "dropout_probability" in result
    assert "risk_level" in result
    assert result["risk_level"] in ("High", "Medium", "Low")
    assert 0.0 <= result["dropout_probability"] <= 1.0


def test_dropout_predict_low_risk_student():
    from backend.modules.dropout.model import predict_dropout
    features = {
        "age": 17, "studytime": 4, "failures": 0, "absences": 1,
        "G1": 19, "G2": 18, "sex": "F", "address": "U",
        "schoolsup": "yes", "famsup": "yes",
        "freetime": 2, "goout": 1, "health": 5, "famrel": 5,
    }
    result = predict_dropout(features)
    # A high-performing student should have low dropout probability
    assert result["dropout_probability"] < 0.6


# ── Wellness tests ────────────────────────────────────────
def test_score_answers_minimal():
    from backend.modules.sentiment.wellness_agent import score_answers
    result = score_answers([0, 0, 0, 0, 0, 0, 0])
    assert result["level"] == "minimal"
    assert result["total_score"] == 0


def test_score_answers_severe():
    from backend.modules.sentiment.wellness_agent import score_answers
    result = score_answers([3, 3, 3, 3, 3, 3, 3])
    assert result["level"] == "severe"


def test_sentiment_analysis():
    from backend.modules.sentiment.wellness_agent import analyze_sentiment_text
    positive = analyze_sentiment_text("I am feeling great and very happy today!")
    negative = analyze_sentiment_text("I feel terrible, hopeless, and depressed.")
    assert positive["label"] == "positive"
    assert negative["label"] == "negative"


def test_crisis_detection():
    from backend.modules.sentiment.wellness_agent import check_crisis
    assert check_crisis("I want to kill myself") is True
    assert check_crisis("I am feeling a bit sad today") is False


# ── Config tests ──────────────────────────────────────────
def test_settings_load():
    from backend.config import get_settings
    s = get_settings()
    assert s.groq_model == "openai/gpt-oss-120b"
    assert s.jwt_algorithm == "HS256"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
