import cv2
import numpy as np
import face_recognition
import pandas as pd

# Load images
img_elon = face_recognition.load_image_file('images/Elon Musk.jpg')
img_elon = cv2.cvtColor(img_elon, cv2.COLOR_BGR2RGB)
img_test = face_recognition.load_image_file('images/Durgesh.jpg')
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

# Find faces and encodings
face_locations_elon = face_recognition.face_locations(img_elon)
face_encodings_elon = face_recognition.face_encodings(img_elon, face_locations_elon)

# Handle potential case of no face detected
if len(face_locations_elon) == 0:
    print("No face found in Elon Musk image.")
else:
    # Get the first face location and encoding (assuming single face)
    face_location_elon = face_locations_elon[0]
    face_encoding_elon = face_encodings_elon[0]

    # Draw rectangle around the face
    cv2.rectangle(img_elon, (face_location_elon[3], face_location_elon[0]),
                  (face_location_elon[1], face_location_elon[2]), (255, 0, 255), 2)

# Repeat for test image
face_locations_test = face_recognition.face_locations(img_test)
face_encodings_test = face_recognition.face_encodings(img_test, face_locations_test)

# Handle potential case of no face detected
if len(face_locations_test) == 0:
    print("No face found in Elon Test image.")
else:
    # Get the first face location and encoding (assuming single face)
    face_location_test = face_locations_test[0]
    face_encoding_test = face_encodings_test[0]

    # Draw rectangle around the face
    cv2.rectangle(img_test, (face_location_test[3], face_location_test[0]),
                  (face_location_test[1], face_location_test[2]), (255, 0, 255), 2)

# Compare faces
results = face_recognition.compare_faces([face_encoding_elon], face_encoding_test)
face_distance = face_recognition.face_distance([face_encoding_elon], face_encoding_test)

# Print results and display on image
print("Is the person the same? ", results)
print("Face distance: ", round(face_distance[0], 2))

# Display text on test image with formatted results
text = f"Same Person: {results[0]} (Distance: {round(face_distance[0], 2)})"
cv2.putText(img_test, text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# Display images
cv2.imshow('Elon Musk', img_elon)
cv2.imshow('Elon Test', img_test)
cv2.waitKey(0)

