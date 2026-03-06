import os
import cv2
import torch
import pickle
import numpy as np
import pandas as pd
import datetime
from PIL import Image
from torchvision import transforms
import face_recognition
import firebase_admin
from firebase_admin import credentials, db
from pymongo import MongoClient
import time

# -----------------------------
# Setup Paths & Working Directory
# -----------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# Firebase Initialization
# -----------------------------
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://face-attendance-10774-default-rtdb.firebaseio.com/"
})

# -----------------------------
# MongoDB Initialization
# -----------------------------
mongo_uri = "mongodb+srv://studentUser:1234@cluster0.zodzonx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)
mongo_db = client["studentdb"]
students_collection = mongo_db["students"]

# -----------------------------
# Device & Anti-Spoofing Model Loader
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "trained_models/antispoof_fullmodels.pkl"

def load_models():
    t0 = time.perf_counter()
    models_dict = torch.load(model_path, map_location=device, weights_only=False)
    for name, model in models_dict.items():
        model.to(device)
        model.eval()
    t1 = time.perf_counter()
    print(f"[TIMING] model load time: {t1 - t0:.4f} sec")
    return models_dict

models_dict = load_models()

# Transform for anti-spoofing input
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def is_real_face(face_img):
    """Returns True if face is real, False if spoof."""
    t0 = time.perf_counter()
    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    probs_sum = None

    with torch.no_grad():
        for name, model in models_dict.items():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs

    avg_probs = probs_sum / len(models_dict)
    pred_class = torch.argmax(avg_probs, dim=1).item()
    t1 = time.perf_counter()

    return (pred_class == 1 , t1 - t0)  # 1 = real, 0 = spoof

# -----------------------------
# Load Face Encodings
# -----------------------------
with open("ENcodedFile.p", "rb") as file:
    encodeAndIdList = pickle.load(file)
encode, studentIDs = encodeAndIdList

# -----------------------------
# Function to Process Group Photo
# -----------------------------
def process_group_photo(group_photo_path):
    results = {"students": [], "timings": {}}

    t_total_start = time.perf_counter()

    if not os.path.exists(group_photo_path):
        return {"status": "error", "message": f"Image not found at {group_photo_path}"}

    t_read_start = time.perf_counter()
    img = cv2.imread(group_photo_path)
    if img is None:
        return {"status": "error", "message": f"Failed to read image at {group_photo_path}"}
    # image resizing + color conversion included in photo processing timing
    img_small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    # face detection & encoding
    face_locations = face_recognition.face_locations(img_small)
    face_encodings = face_recognition.face_encodings(img_small, face_locations)
    t_read_end = time.perf_counter()

    photo_processing_time = t_read_end - t_read_start
    results["timings"]["photo_processing_time"] = photo_processing_time
    results["timings"]["num_faces_detected"] = len(face_encodings)

    if not face_encodings:
        t_total_end = time.perf_counter()
        results["timings"]["total_time"] = t_total_end - t_total_start
        return {"status": "error", "message": "No faces detected in the group photo.", "timings": results["timings"]}

    # loop through faces and measure per-face steps
    per_face_timings = []
    for encoding, location in zip(face_encodings, face_locations):
        face_t0 = time.perf_counter()
        student_result = {"student_id": None, "status": None, "details": {}}
        # recognition (matching)
        t_recog_start = time.perf_counter()
        matches = face_recognition.compare_faces(encode, encoding)
        face_distances = face_recognition.face_distance(encode, encoding)
        match_index = np.argmin(face_distances)
        t_recog_end = time.perf_counter()

        if matches[match_index]:
            student_id = studentIDs[match_index]
            student_result["student_id"] = student_id
            print(f"Match found for student ID: {student_id}")

            # Crop face for anti-spoofing (scale back to original)
            top, right, bottom, left = [v * 4 for v in location]
            face_crop = img[top:bottom, left:right]

            # ---------------- Anti-spoofing (measured inside function) ----------------
            is_real, antispoof_time = is_real_face(face_crop)
            if not is_real:
                msg = f"Spoof detected for student {student_id}. Attendance NOT marked."
                print(msg)
                student_result["status"] = "spoof"
                student_result["details"] = {"message": msg}
                per_face_timings.append({
                    "student_id": student_id,
                    "recognition_time": t_recog_end - t_recog_start,
                    "antispoof_time": antispoof_time,
                    "db_update_time": 0.0,
                    "total_face_time": time.perf_counter() - face_t0
                })
                results["students"].append(student_result)
                continue

            # ---------------- Database updates (Firebase + Mongo + Excel) ----------------
            t_db_start = time.perf_counter()

            student_info = db.reference(f"Students/{student_id}").get()
            if student_info is None:
                msg = f"Student ID {student_id} not found in Firebase."
                print(msg)
                student_result["status"] = "not_found"
                student_result["details"] = {"message": msg}
                db_update_time = time.perf_counter() - t_db_start
                per_face_timings.append({
                    "student_id": student_id,
                    "recognition_time": t_recog_end - t_recog_start,
                    "antispoof_time": antispoof_time,
                    "db_update_time": db_update_time,
                    "total_face_time": time.perf_counter() - face_t0
                })
                results["students"].append(student_result)
                continue

            # timestamp check + firebase update
            last_attendance = student_info.get("last_attendence_date", "2000-01-01 00:00:00")
            last_attendance_dt = datetime.datetime.strptime(last_attendance, "%Y-%m-%d %H:%M:%S")
            time_diff = (datetime.datetime.now() - last_attendance_dt).total_seconds()

            if time_diff > 10:
                student_info["total_attendence"] = student_info.get("total_attendence", 0) + 1
                student_info["last_attendence_date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                db.reference(f"Students/{student_id}").set(student_info)
                print(f"[Firebase] Attendance updated for {student_info.get('name', student_id)}.")

                # # MongoDB Update
                # mongo_student = students_collection.find_one({"rollNumber": student_id})
                # if mongo_student:
                #     new_present_count = mongo_student.get("presentCount", 0) + 1
                #     new_total_count = mongo_student.get("totalClassCount", 0) + 1
                #     new_percentage = (new_present_count / new_total_count) * 100 if new_total_count > 0 else 0.0

                #     students_collection.update_one(
                #         {"rollNumber": student_id},
                #         {
                #             "$set": {
                #                 "presentCount": new_present_count,
                #                 "totalClassCount": new_total_count,
                #                 "attendancePercentage": new_percentage,
                #                 "updatedAt": datetime.datetime.now().isoformat()
                #             }
                #         }
                #     )
                #     print(f"[MongoDB] Attendance updated for rollNumber {student_id}.")
                #     student_result["status"] = "marked"
                #     student_result["details"] = {
                #         "name": student_info.get("name", student_id),
                #         "presentCount": new_present_count,
                #         "totalClassCount": new_total_count,
                #         "attendancePercentage": new_percentage
                #     }
                # else:
                #     msg = f"No MongoDB record found for rollNumber {student_id}."
                #     print(msg)
                #     student_result["status"] = "mongo_missing"
                #     student_result["details"] = {"message": msg}

                # DataLogger (Excel)
                if os.path.exists("datalogger.xlsx"):
                    existing_df = pd.read_excel("datalogger.xlsx", sheet_name="Sheet1")
                else:
                    existing_df = pd.DataFrame(columns=["Student_IDs", "Names", "Attendence_count", "Attempt_time"])

                new_data = {
                    "Student_IDs": [student_id],
                    "Names": [student_info.get("name", student_id)],
                    "Attendence_count": [student_info["total_attendence"]],
                    "Attempt_time": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                }
                new_df = pd.DataFrame(new_data)
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)

                with pd.ExcelWriter("datalogger.xlsx", mode="w", engine="openpyxl") as writer_file:
                    updated_df.to_excel(writer_file, sheet_name="Sheet1", index=False)

            t_db_end = time.perf_counter()
            db_update_time = t_db_end - t_db_start

            # record per-face timings
            per_face_timings.append({
                "student_id": student_id,
                "recognition_time": t_recog_end - t_recog_start,
                "antispoof_time": antispoof_time,
                "db_update_time": db_update_time,
                "total_face_time": time.perf_counter() - face_t0
            })

        else:
            # no match case
            student_result["status"] = "no_match"
            student_result["details"] = {"message": "No matching face found in encodings."}
            per_face_timings.append({
                "student_id": None,
                "recognition_time": t_recog_end - t_recog_start,
                "antispoof_time": 0.0,
                "db_update_time": 0.0,
                "total_face_time": time.perf_counter() - face_t0
            })

        results["students"].append(student_result)

    t_total_end = time.perf_counter()
    results["timings"]["per_face"] = per_face_timings
    results["timings"]["total_time"] = t_total_end - t_total_start

    results["status"] = "success"
    results["message"] = f"Processed {len(results['students'])} students from {os.path.basename(group_photo_path)}"
    return results

# -----------------------------
# Main Callable Function for Flask
# -----------------------------
def run_model(image_path):

    try:
        return process_group_photo(image_path)
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -----------------------------
# Debug Run
# -----------------------------
if __name__ == "__main__":
    test_path = "C:\\Users\\Asus\\Desktop\\VS CODES\\Attttt\\AttendenceProject\\group_photo_sample.jpg"
    result = run_model(test_path)
    print(result)