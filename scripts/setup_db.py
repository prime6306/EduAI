"""
scripts/setup_db.py
Creates all MongoDB indexes and seeds initial student data.
Run once after setting up the database:
    python scripts/setup_db.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db.mongodb import get_sync_db
from datetime import datetime

def setup_indexes():
    db = get_sync_db()

    print("Creating indexes...")

    # Users
    db.users.create_index("email", unique=True)
    db.users.create_index("student_id")

    # Students (attendance)
    db.students.create_index("student_id", unique=True)
    db.attendance_logs.create_index([("student_id", 1), ("timestamp", -1)])
    db.attendance_logs.create_index("timestamp")

    # QA history
    db.qa_history.create_index([("user_id", 1), ("timestamp", -1)])
    db.qa_history.create_index("pdf_id")

    # PDFs
    db.pdfs.create_index([("user_id", 1), ("pdf_id", 1)])
    db.pdfs.create_index("content_hash")

    # NLP pipelines
    db.topic_pipelines.create_index([("user_id", 1), ("created_at", -1)])
    db.topic_pipelines.create_index("topic")

    # Quiz results
    db.quiz_results.create_index([("user_id", 1), ("timestamp", -1)])

    # Dropout predictions
    db.dropout_predictions.create_index("timestamp")

    # Wellness sessions
    db.wellness_sessions.create_index("user_id")
    db.wellness_sessions.create_index("session_id", unique=True)

    # Plagiarism reports
    db.plagiarism_reports.create_index("created_at")

    print("✅ All indexes created.")


def seed_students():
    """Seed demo students from the original database.py."""
    db = get_sync_db()
    students = [
        {"student_id": "101", "name": "Pranjal Mishra",        "branch": "ECE", "year": "3rd year", "total_attendance": 0},
        {"student_id": "102", "name": "Vedath Batham",         "branch": "ECE", "year": "3rd year", "total_attendance": 0},
        {"student_id": "103", "name": "Priyanshu Kashyap",     "branch": "ECE", "year": "3rd year", "total_attendance": 0},
        {"student_id": "104", "name": "Shristi Dixit",         "branch": "ECE", "year": "3rd year", "total_attendance": 0},
        {"student_id": "105", "name": "Anjali Sharma",         "branch": "ECE", "year": "3rd year", "total_attendance": 0},
        {"student_id": "106", "name": "Akarsh Tiwari",         "branch": "ECE", "year": "3rd year", "total_attendance": 0},
        {"student_id": "107", "name": "Divyanshi Sonkar",      "branch": "ECE", "year": "3rd year", "total_attendance": 0},
        {"student_id": "108", "name": "Priyanshu Chaudhary",   "branch": "ECE", "year": "3rd year", "total_attendance": 0},
    ]
    inserted = 0
    for s in students:
        if not db.students.find_one({"student_id": s["student_id"]}):
            s["last_attendance_date"] = "2000-01-01 00:00:00"
            s["created_at"] = datetime.utcnow()
            db.students.insert_one(s)
            inserted += 1
    print(f"✅ Seeded {inserted} students (skipped {len(students)-inserted} existing).")


if __name__ == "__main__":
    setup_indexes()
    seed_students()
    print("\n🎉 Database setup complete!")
