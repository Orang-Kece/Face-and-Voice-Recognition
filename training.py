# Mengimport package yang diperlukan
import cv2
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Membuat variabel recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Untuk detector menggunakan file haarcascade_frontalface_default.xml
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Membuat fungsi dengan getImagesWithLabels parameter path
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

# Load data dari folder Dataset
faces, Ids = getImagesWithLabels('Dataset')

if faces:
    # Membagi data menjadi training set dan test set
    x_train, x_test, y_train, y_test = train_test_split(faces, Ids, test_size=0.2, random_state=42)

    # Melatih model menggunakan training set
    recognizer.train(x_train, np.array(y_train))
    print("Model training selesai.")

    # Menyimpan model yang telah dilatih
    recognizer.save('Dataset/training.xml')
    print("Model disimpan di Dataset/training.xml")

    # Menguji model menggunakan test set dengan confidence evaluation
    predictions = []
    confidences = []
    for test_face in x_test:
        label, confidence = recognizer.predict(test_face)
        predictions.append(label)
        confidences.append(confidence)

    # Menghitung akurasi tanpa filtering confidence
    overall_accuracy = accuracy_score(y_test, predictions) * 100
    print(f"Akurasi model tanpa filtering: {overall_accuracy:.2f}%")

    # Menghitung akurasi dengan filtering berdasarkan confidence
    threshold = 50  # Confidence threshold (tune this value as needed)
    filtered_predictions = []
    filtered_y_test = []

    for i, confidence in enumerate(confidences):
        if confidence < threshold:
            filtered_predictions.append(predictions[i])
            filtered_y_test.append(y_test[i])

    if filtered_predictions:
        filtered_accuracy = accuracy_score(filtered_y_test, filtered_predictions) * 100
        print(f"Akurasi model setelah filtering berdasarkan confidence (<{threshold}): {filtered_accuracy:.2f}%")
    else:
        print("Tidak ada prediksi yang memenuhi ambang batas confidence.")

    # Menampilkan rata-rata confidence
    average_confidence = np.mean(confidences)
    print(f"Rata-rata confidence: {average_confidence:.2f}")
else:
    print("Tidak ada data wajah yang ditemukan untuk pelatihan.")
