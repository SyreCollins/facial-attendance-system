import cv2
import os
import json
import numpy as np
from datetime import datetime
import mediapipe as mp

# File paths
ATTENDANCE_FILE = "attendance.txt"
DATASET_DIR = "dataset"
LABEL_MAP_FILE = "label_map.json"

# Ensure necessary directories and files exist
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w") as f:
        f.write("Name,Timestamp\n")  # Add headers to the file

# Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Save attendance to the text file
def log_attendance(name):
    with open(ATTENDANCE_FILE, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{name},{timestamp}\n")

# Train the recognizer
def train_recognizer():
    label_map = {}
    label_counter = 0

    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        if os.path.isdir(person_path):
            label_map[label_counter] = person_name  # Map label ID to person name
            label_counter += 1

    # Save the label map
    with open(LABEL_MAP_FILE, "w") as f:
        json.dump(label_map, f)

    print("Training completed. Label map saved!")

# Snapshot-based recognition
def snapshot_recognition():
    if not os.path.exists(LABEL_MAP_FILE):
        print("Recognizer not trained. Please capture face data first!")
        return

    # Load the label map
    with open(LABEL_MAP_FILE, "r") as f:
        label_map = json.load(f)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    training_images = []
    labels = []

    for label, person_name in label_map.items():
        person_dir = os.path.join(DATASET_DIR, person_name)
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                training_images.append(img)
                labels.append(int(label))

    face_recognizer.train(training_images, np.array(labels))

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

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                    cv2.putText(frame, "Align your face properly", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Snapshot Recognition", frame)

            # Press 'C' to capture and recognize
            if cv2.waitKey(1) & 0xFF == ord("c"):
                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                        face = frame[y:y + h_box, x:x + w_box]
                        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                        # Resize the face for recognition
                        gray_face = cv2.resize(gray_face, (200, 200))

                        try:
                            label, confidence = face_recognizer.predict(gray_face)
                            if confidence < 100:  # Confidence threshold
                                name = label_map[str(label)]
                                log_attendance(name)
                                print(f"Recognized {name} with confidence: {confidence:.2f}")
                            else:
                                name = "Unknown"
                                print("No match found in the dataset.")

                        except Exception as e:
                            print(f"Error in face recognition: {e}")

                break

            # Press 'Q' to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cam.release()
    cv2.destroyAllWindows()

# Capture face images
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
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f"{person_dir}/{count}.jpg", gray_frame)
            print(f"Captured {count} image(s) for {name}")

        if count >= 20 or (cv2.waitKey(1) & 0xFF == ord("q")):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Face data for {name} captured successfully!")

    train_recognizer()

# Main menu
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
