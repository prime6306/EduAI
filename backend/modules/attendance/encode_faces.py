"""
backend/modules/attendance/encode_faces.py
Regenerate ENcodedFile.p from Images/ folder.
Run whenever new student photos are added:
    python backend/modules/attendance/encode_faces.py
"""
import cv2
import face_recognition
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from backend.config import get_settings

def encode_faces(images_dir: str = "models/Images", output_path: str = None):
    if output_path is None:
        output_path = get_settings().face_encodings_path

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"Created {images_dir}/ — add student photos named <student_id>.jpg")
        return

    path_list = [p for p in os.listdir(images_dir) if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not path_list:
        print(f"No images found in {images_dir}/")
        return

    img_list, student_ids = [], []
    for path in path_list:
        img = cv2.imread(os.path.join(images_dir, path))
        if img is not None:
            img_list.append(img)
            student_ids.append(os.path.splitext(path)[0])

    print(f"Found {len(img_list)} images: {student_ids}")

    encodings = []
    for img in img_list:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        found = face_recognition.face_encodings(rgb)
        if found:
            encodings.append(found[0])
        else:
            print("⚠️  No face found in one image — skipping")
            student_ids.pop(len(encodings))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump([encodings, student_ids], f)

    print(f"✅ Encoded {len(encodings)} faces → {output_path}")

if __name__ == "__main__":
    encode_faces()
