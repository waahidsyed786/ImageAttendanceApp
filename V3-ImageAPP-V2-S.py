import sys
import os
import cv2
import face_recognition
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, 
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView
)
from datetime import datetime

class AttendanceApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Recognition Attendance Tracker")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout(self)

        # Button to load roll numbers
        self.roll_button = QPushButton("Load Roll Numbers CSV")
        self.roll_button.clicked.connect(self.load_roll_numbers)
        self.layout.addWidget(self.roll_button)

        # Button to load reference images
        self.image_folder_button = QPushButton("Load Reference Images Folder")
        self.image_folder_button.clicked.connect(self.load_reference_images)
        self.layout.addWidget(self.image_folder_button)

        # Button to load group image
        self.image_button = QPushButton("Load Group Image")
        self.image_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.image_button)

        # Button to update attendance
        self.update_button = QPushButton("Update Attendance")
        self.update_button.clicked.connect(self.update_attendance)
        self.layout.addWidget(self.update_button)

        # Label to show status
        self.status_label = QLabel("Status: Ready")
        self.layout.addWidget(self.status_label)

        # Table to display detected faces and attendance
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Roll Number", "Attendance"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.table)

        # Button to save attendance
        self.save_button = QPushButton("Save Attendance")
        self.save_button.clicked.connect(self.save_attendance)
        self.layout.addWidget(self.save_button)

        # Data structures to hold roll numbers and face encodings
        self.roll_numbers = []  # List of roll numbers
        self.known_face_encodings = []  # Store known face encodings
        self.known_face_roll_numbers = []  # Map face encodings to roll numbers
        self.group_face_encodings = []  # Detected faces in group image

    def load_roll_numbers(self):
        """Load the CSV file containing roll numbers."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Roll Numbers CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            try:
                df = pd.read_csv(file_path)
                if 'Roll Number' not in df.columns:
                    self.status_label.setText("CSV must contain 'Roll Number' column.")
                    return
                self.roll_numbers = df['Roll Number'].astype(str).tolist()
                self.status_label.setText(f"Loaded {len(self.roll_numbers)} roll numbers.")

                # Populate the table with roll numbers and default attendance as "A"
                self.table.setRowCount(0)
                for roll_number in self.roll_numbers:
                    row_position = self.table.rowCount()
                    self.table.insertRow(row_position)
                    self.table.setItem(row_position, 0, QTableWidgetItem(roll_number))
                    self.table.setItem(row_position, 1, QTableWidgetItem("A"))  # Default to Absent

            except Exception as e:
                self.status_label.setText(f"Error loading CSV: {str(e)}")

    def load_reference_images(self):
        """Load reference images from a folder and extract face encodings."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Reference Images Folder")
        if folder_path:
            self.known_face_encodings = []
            self.known_face_roll_numbers = []

            loaded_count = 0
            for roll_number in self.roll_numbers:
                # Attempt to find image with roll number as filename (e.g., 1001.jpg)
                # Support multiple extensions
                found = False
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_path = os.path.join(folder_path, f"{roll_number}{ext}")
                    if os.path.exists(image_path):
                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            self.known_face_encodings.append(encodings[0])
                            self.known_face_roll_numbers.append(roll_number)
                            loaded_count += 1
                            self.status_label.setText(f"Loaded encoding for Roll Number: {roll_number}")
                        else:
                            self.status_label.setText(f"No face found in image: {image_path}")
                        found = True
                        break  # Stop searching extensions once found
                if not found:
                    self.status_label.setText(f"No image found for Roll Number: {roll_number}")

            self.status_label.setText(f"Loaded {loaded_count} face encodings from reference images.")

    def load_image(self):
        """Load an image containing a group of students' faces."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Group Image File", "", "Image Files (*.jpg *.png *.jpeg);;All Files (*)"
        )
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        """Process the group image to detect and recognize faces."""
        image = cv2.imread(file_path)
        if image is None:
            self.status_label.setText("Failed to load image.")
            return

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces and get their encodings
        face_locations = face_recognition.face_locations(rgb_image)
        detected_face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        if len(detected_face_encodings) == 0:
            self.status_label.setText("No faces detected in the image.")
            return

        self.group_face_encodings = detected_face_encodings  # Store for update

        # Match detected faces with known encodings
        present_roll_numbers = set()
        for detected_encoding in detected_face_encodings:
            matches = face_recognition.compare_faces(
                self.known_face_encodings, detected_encoding, tolerance=0.5
            )  # Lower tolerance for stricter matching
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, detected_encoding
            )
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1
            if best_match_index != -1 and matches[best_match_index]:
                matched_roll_number = self.known_face_roll_numbers[best_match_index]
                present_roll_numbers.add(matched_roll_number)
                self.status_label.setText(f"Detected and matched Roll Number: {matched_roll_number}")
            else:
                self.status_label.setText("A detected face did not match any known roll numbers.")

        # Automatically mark attendance
        for row in range(self.table.rowCount()):
            roll_number = self.table.item(row, 0).text()
            if roll_number in present_roll_numbers:
                self.table.setItem(row, 1, QTableWidgetItem("P"))  # Mark Present
            else:
                self.table.setItem(row, 1, QTableWidgetItem("A"))  # Mark Absent

        self.status_label.setText(f"Processed image. Detected {len(detected_face_encodings)} face(s).")

    def update_attendance(self):
        """Update the attendance based on previously loaded group image."""
        if not self.group_face_encodings:
            self.status_label.setText("No group image processed yet.")
            return

        # Match detected faces with known encodings
        present_roll_numbers = set()
        for detected_encoding in self.group_face_encodings:
            matches = face_recognition.compare_faces(
                self.known_face_encodings, detected_encoding, tolerance=0.5
            )  # Adjust tolerance as needed
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, detected_encoding
            )
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1
            if best_match_index != -1 and matches[best_match_index]:
                matched_roll_number = self.known_face_roll_numbers[best_match_index]
                present_roll_numbers.add(matched_roll_number)
                self.status_label.setText(f"Updated and marked Present: {matched_roll_number}")
            else:
                self.status_label.setText("A detected face did not match any known roll numbers.")

        # Automatically mark attendance
        for row in range(self.table.rowCount()):
            roll_number = self.table.item(row, 0).text()
            if roll_number in present_roll_numbers:
                self.table.setItem(row, 1, QTableWidgetItem("P"))  # Mark Present
            else:
                self.table.setItem(row, 1, QTableWidgetItem("A"))  # Mark Absent

        self.status_label.setText(f"Attendance updated based on processed group image.")

    def save_attendance(self):
        """Save the attendance to a CSV file."""
        if len(self.roll_numbers) == 0:
            self.status_label.setText("No roll numbers loaded. Please load the roll numbers CSV first.")
            return

        selected_date = datetime.now().strftime("%Y-%m-%d")
        new_filename = f"attendance_{selected_date}.csv"

        # Prepare attendance data
        attendance_data = []
        for row in range(self.table.rowCount()):
            roll_number = self.table.item(row, 0).text()
            attendance_status = self.table.item(row, 1).text()
            attendance_data.append([roll_number, attendance_status])

        # Save attendance data to CSV
        try:
            pd.DataFrame(attendance_data, columns=["Roll Number", "Attendance"]).to_csv(new_filename, index=False)
            self.status_label.setText(f"Attendance saved to {new_filename}")
        except Exception as e:
            self.status_label.setText(f"Error saving attendance: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AttendanceApp()
    window.show()
    sys.exit(app.exec_())
