import cv2
import os
import json
import numpy as np
from datetime import datetime

# File paths
ATTENDANCE_FILE = "attendance.txt"
DATASET_DIR = "dataset"
MODEL_FILE = "face_recognizer.yml"
LABEL_MAP_FILE = "label_map.json"

# Ensure necessary directories and files exist
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w") as f:
        f.write("Name,Timestamp\n")  # Add headers to the file

# Save attendance to the text file
def log_attendance(name):
    with open(ATTENDANCE_FILE, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{name},{timestamp}\n")

# Augment face images for training
def augment_image(img):
    flipped_img = cv2.flip(img, 1)  # Horizontal flip
    equalized_img = cv2.equalizeHist(img)  # Histogram equalization
    return [img, flipped_img, equalized_img]

# Train the face recognizer
def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_map = {}
    label_counter = 0
    image_size = (100, 100)  # Ensure all images are resized to this dimension

    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        if os.path.isdir(person_path):
            label_map[label_counter] = person_name  # Map label ID to person name
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, image_size)  # Resize to consistent dimensions
                    augmented_imgs = augment_image(img)  # Augment images
                    faces.extend(augmented_imgs)
                    labels.extend([label_counter] * len(augmented_imgs))
            label_counter += 1

    # Convert lists to numpy arrays
    faces = np.array(faces, dtype="uint8")
    labels = np.array(labels, dtype="int32")

    # Train the recognizer
    recognizer.train(faces, labels)
    recognizer.write(MODEL_FILE)  # Save the trained model

    # Save the label map
    with open(LABEL_MAP_FILE, "w") as f:
        json.dump(label_map, f)

    print("Training completed. Model and label map saved!")

# Real-time facial recognition
def real_time_recognition():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABEL_MAP_FILE):
        print("Recognizer not trained. Please capture face data first!")
        return

    # Load the label map
    with open(LABEL_MAP_FILE, "r") as f:
        label_map = json.load(f)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    attendance = []

    print("Press 'Q' to quit.")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to access the camera!")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=6, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face = gray_frame[y:y + h, x:x + w]
            face = cv2.resize(face, (100, 100))  # Ensure consistent size
            face = cv2.equalizeHist(face)  # Normalize face before recognition
            label, confidence = recognizer.predict(face)
            print(f"Detected {label_map.get(str(label), 'Unknown')} with confidence: {confidence}")

            if confidence < 85:  # Adjusted threshold
                name = label_map.get(str(label), "Unknown")
                if name not in attendance:
                    attendance.append(name)
                    log_attendance(name)
                    print(f"Attendance marked for {name}")

                color = (0, 255, 0)  # Green for recognized faces
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for unrecognized faces

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name} ({int(confidence)}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Real-Time Recognition", frame)

        # Press 'Q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

# Capture and store face images
def capture_face(name):
    os.makedirs(f"{DATASET_DIR}/{name}", exist_ok=True)
    cam = cv2.VideoCapture(0)
    count = 0

    print("Press 'C' to capture images. Press 'Q' to quit.")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to access the camera!")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Capture Face", frame)

        # Press 'C' to capture images
        if cv2.waitKey(1) & 0xFF == ord("c"):
            count += 1
            cv2.imwrite(f"{DATASET_DIR}/{name}/{count}.jpg", gray_frame)
            print(f"Captured {count} image(s) for {name}")

        # Press 'Q' to quit after capturing
        if count >= 20 or (cv2.waitKey(1) & 0xFF == ord("q")):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Face data for {name} captured successfully!")

    # Train the recognizer automatically after capturing face data
    train_recognizer()

# View attendance records
def view_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            records = f.readlines()
        if len(records) > 1:
            print("Attendance Records:")
            for line in records[1:]:
                print(line.strip())
        else:
            print("No attendance records found.")
    else:
        print("Attendance file not found.")

# Main menu
def main():
    while True:
        print("\nFacial Recognition Attendance System")
        print("1. Capture Face")
        print("2. Real-Time Recognition")
        print("3. View Attendance")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            name = input("Enter the name of the person: ")
            if name:
                capture_face(name)
            else:
                print("Name cannot be empty.")
        elif choice == "2":
            real_time_recognition()
        elif choice == "3":
            view_attendance()
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
