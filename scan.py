import cv2, time
import os
from PIL import Image

camera = 0  # Webcam default (change if using external)
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Dataset/training.xml')

a = 0
id_map = {
    1: 'Weldon', 2: 'Elbert', 3: 'Zoe', 4: 'Kenneth', 5: 'Rayhan',
    6: 'Alfredo', 7: 'Bryan', 8: 'Janssen', 9: 'Gilbert', 10: 'Lovina',
    11: 'Jean', 12: 'Hugo', 13: 'Bagas', 14: 'Flo', 15: 'Jerremy',
    16: 'Reuben', 17: 'Caroline'
}

# Function to get face name and confidence
def get_face_name_and_confidence(id, conf):
    if conf < 50:  # If confidence is lower than 50, consider it 'Unknown'
        return "Unknown", conf
    name = id_map.get(id, "Unknown")  # Get the name from the id_map or return 'Unknown' if id not found
    return name, conf

while True:
    a += 1
    check, frame = video.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = faceDeteksi.detectMultiScale(abu, 1.3, 5)

    for (x, y, w, h) in wajah:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, conf = recognizer.predict(abu[y:y + h, x:x + w])

        # Get name and confidence
        name, confidence = get_face_name_and_confidence(id, conf)
        
        # Display name and confidence next to the face
        cv2.putText(frame, f"{name} - {confidence:.2f}", (x + 40, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):  # Press 'q' to quit
        break

video.release()
cv2.destroyAllWindows()
