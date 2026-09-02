#!/usr/bin/env python
"""
CLI for student registration and face enrollment, outside the web UI.

Usage:
    python scripts/add_student.py --id 109 --name "Rahul Verma" --photo photo.jpg
    python scripts/add_student.py --list
    python scripts/add_student.py --list-faces
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app  # noqa: E402
from app.modules.attendance import enrollment_service, face_engine  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="EduAI student management CLI")
    parser.add_argument("--id", help="Student ID")
    parser.add_argument("--name", help="Full name")
    parser.add_argument("--branch", default="", help="Branch (e.g. CS)")
    parser.add_argument("--year", default="", help="Year (e.g. 3)")
    parser.add_argument("--email", default="", help="Email (optional)")
    parser.add_argument("--photo", help="Path to a front-facing photo for face enrollment")
    parser.add_argument("--list", action="store_true", help="List all registered students")
    parser.add_argument("--list-faces", action="store_true", help="List students with an enrolled face")
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        if args.list:
            students = enrollment_service.list_students()
            for s in students:
                print(f"{s['student_id']:<10} {s['name']:<30} {s.get('branch',''):<6} "
                      f"{s.get('year',''):<4} attendance={s.get('total_attendance',0)}")
            print(f"\n{len(students)} student(s) total.")
            return

        if args.list_faces:
            face_engine.load_face_encodings(app)
            cache = face_engine.get_cache()
            for sid in cache["student_ids"]:
                print(sid)
            print(f"\n{len(cache['student_ids'])} face(s) enrolled.")
            return

        if not args.id or not args.name:
            parser.error("--id and --name are required to add a student (or use --list / --list-faces)")

        try:
            enrollment_service.add_student(args.id, args.name, args.branch, args.year, args.email)
            print(f"Added student {args.id} - {args.name}")
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

        if args.photo:
            try:
                enrollment_service.enroll_face(app, args.id, args.photo)
                print(f"Face enrolled for {args.id}")
            except ValueError as exc:
                print(f"Face enrollment failed: {exc}")
                sys.exit(1)


if __name__ == "__main__":
    main()
