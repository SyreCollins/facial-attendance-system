import cv2
import os
import json
import numpy as np
from datetime import datetime
import mediapipe as mp

# File paths
ATTENDANCE_DIR = "attendance_records"
DATASET_DIR = "dataset"
LABEL_MAP_FILE = "label_map.json"

# Ensure necessary directories and files exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def log_attendance(name):
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(ATTENDANCE_DIR, f"{date_str}.txt")
    with open(filename, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{name},{timestamp}\n")

def preprocess_face(face_image):
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    filtered = cv2.bilateralFilter(blurred, 9, 75, 75)
    return filtered

def train_recognizer():
    label_map = {}
    label_counter = 0
    training_images = []
    labels = []
    
    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        if os.path.isdir(person_path):
            label_map[label_counter] = person_name
            for image_name in os.listdir(person_path):
                img_path = os.path.join(person_path, image_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    training_images.append(img)
                    labels.append(label_counter)
            label_counter += 1
    
    with open(LABEL_MAP_FILE, "w") as f:
        json.dump(label_map, f)
    
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_recognizer.train(training_images, np.array(labels))
    face_recognizer.write("trained_model.xml")
    print("Training completed. Model saved!")

def capture_face(name):
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    cam = cv2.VideoCapture(0)
    count = 0
    
    print("Press 'C' to capture images. Press 'Q' to quit.")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to access the camera!")
            break
        
        cv2.imshow("Capture Face", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("c"):
            count += 1
            gray_frame = preprocess_face(frame)
            cv2.imwrite(f"{person_dir}/{count}.jpg", gray_frame)
            print(f"Captured {count} image(s) for {name}")
        
        if count >= 20 or (cv2.waitKey(1) & 0xFF == ord("q")):
            break
    
    cam.release()
    cv2.destroyAllWindows()
    print(f"Face data for {name} captured successfully!")
    train_recognizer()

def snapshot_recognition():
    if not os.path.exists("trained_model.xml") or not os.path.exists(LABEL_MAP_FILE):
        print("Recognizer not trained. Please capture face data first!")
        return
    
    with open(LABEL_MAP_FILE, "r") as f:
        label_map = json.load(f)
    
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_recognizer.read("trained_model.xml")
    
    cam = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        print("Press 'C' to capture a snapshot. Press 'Q' to quit.")
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to access the camera!")
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)
            
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                    
                    face = frame[y:y + h_box, x:x + w_box]
                    gray_face = preprocess_face(face)
                    
                    label, confidence = face_recognizer.predict(gray_face)
                    base_threshold = 70
                    dynamic_threshold = base_threshold + (len(label_map) * 0.5)
                    
                    if confidence < dynamic_threshold:
                        name = label_map[str(label)]
                        log_attendance(name)
                        print(f"Recognized {name} with confidence: {confidence:.2f}")
                    else:
                        print("No match found in the dataset.")
                    
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    cam.release()
    cv2.destroyAllWindows()

def main():
    while True:
        print("\nFacial Recognition Attendance System")
        print("1. Capture Face")
        print("2. Snapshot Recognition")
        print("3. Exit")
        choice = input("Enter your choice: ")
        
        if choice == "1":
            name = input("Enter the name of the person: ")
            if name:
                capture_face(name)
            else:
                print("Name cannot be empty.")
        elif choice == "2":
            snapshot_recognition()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
