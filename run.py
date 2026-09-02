"""
Entry point: `python run.py`
For production, prefer: gunicorn -w 4 -b 0.0.0.0:5000 "run:app"
"""
# Eagerly import dlib and face_recognition on Windows before any other packages
# to prevent "DLL initialization routine failed" (error 1114) from OpenMP runtime clashing
try:
    import dlib  # noqa: F401
    import face_recognition  # noqa: F401
except Exception:
    pass

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=app.config["PORT"], debug=app.config["DEBUG"])
