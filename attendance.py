import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta

# Load images and class names
path = 'images'
classNames = [os.path.splitext(cl)[0] for cl in os.listdir(path)]
images = [cv2.imread(os.path.join(path, cl)) for cl in os.listdir(path)]

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
        except IndexError:
            print("No face found in image")
            encode = None
        encodeList.append(encode)
    return encodeList

def markAttendance(name, attendance_file, last_recorded_time, face_recorded):
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')

    # Check if the face has already been marked today
    if not face_recorded:
        with open(attendance_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if name in line:
                    print("Face already recorded today")
                    return now, face_recorded  # Update last recorded time and flag

    # Check if enough time has passed since the last recording
    if last_recorded_time is None or (now - last_recorded_time) >= timedelta(seconds=10):
        # Write new data to CSV
        with open(attendance_file, 'a') as f:
            f.write(f'\n{name},{dtString}')
        print("Attendance recorded")
        return now, True  # Update last recorded time and set flag to True

    return last_recorded_time, face_recorded  # Return unchanged last recorded time and flag

attendance_file = 'attendance.csv'
encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
last_recorded_time = None  # Initialize last recorded time
face_recorded = False  # Initialize flag for face recording

while True:
    # Capture frame-by-frame
    success, img = cap.read()
    if not success:
        print("Failed to capture frame from webcam")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceloc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            last_recorded_time, face_recorded = markAttendance(name, attendance_file, last_recorded_time, face_recorded)

            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)

    # Check if user pressed 'q', if so, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
