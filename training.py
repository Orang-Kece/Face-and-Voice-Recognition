# Mengimport package yang diperlukan
import cv2, os
import numpy as np
from PIL import Image
# Membuat variabel recognizer

recognizer = cv2.face.LBPHFaceRecognizer_create()
# Untuk detector menggunakan file haarcascade_frontalface_default.xml
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# Membuat fungsi dengan  getImagesWithLabels parameter path
def getImagesWithLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    faceSamples = []
    Ids = []

    for imagePath in imagePaths:
        try:
            # Membuka gambar
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            # Mendapatkan ID dari nama file
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(imageNp)

            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y+h, x:x+w])
                Ids.append(Id)
        except Exception as e:
            print(f"Error processing file {imagePath}: {e}")

    return faceSamples, Ids

faces, Ids = getImagesWithLabels('Dataset')
recognizer.train(faces, np.array(Ids))

# Data training disimpan di folder Dataset dengan nama file training.xml
recognizer.save('Dataset/training.xml')