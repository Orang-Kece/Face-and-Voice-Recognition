# Mengimpor library yang dibutuhkan
import cv2, time  # cv2 untuk OpenCV, time untuk penanganan waktu
import cv2, os  # cv2 untuk OpenCV, os untuk pengelolaan file dan direktori
import numpy as np  # np untuk operasi array numerik
from PIL import Image  # Image untuk memproses gambar dengan PIL (Python Imaging Library)

camera = 0  # Menentukan sumber kamera (0 berarti webcam pertama)
# Membuka webcam menggunakan OpenCV
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)  # Menggunakan DirectShow pada Windows untuk membuka webcam
# Menggunakan classifier untuk mendeteksi wajah dengan algoritma Haar Cascade
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # File classifier untuk deteksi wajah frontal

id = input('Id : ')  # Meminta pengguna untuk memasukkan ID untuk penamaan file gambar
a = 0  # Variabel untuk menghitung jumlah gambar yang diambil

# Membuat model pengenalan wajah dengan algoritma LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()  # Algoritma Local Binary Patterns Histograms (LBPH)
# Untuk deteksi wajah menggunakan file haarcascade_frontalface_default.xml
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # Detektor wajah

# Menggunakan loop untuk mengambil gambar secara terus-menerus
while True:
    a = a + 1  # Increment counter untuk gambar
    check, frame = video.read()  # Membaca frame dari webcam
    # Mengonversi gambar menjadi grayscale (abu-abu) untuk deteksi wajah
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mendeteksi wajah dalam gambar grayscale
    wajah = faceDeteksi.detectMultiScale(abu, 1.3, 5)  # Mendeteksi wajah pada berbagai ukuran
    print(wajah)  # Mencetak hasil deteksi wajah ke terminal
    # Menggambar kotak di sekitar wajah yang terdeteksi
    for(x, y, w, h) in wajah:
        # Menyimpan gambar wajah yang terdeteksi ke dalam folder Dataset dengan format penamaan tertentu
        cv2.imwrite('Dataset/User.'+str(id)+'.'+str(a)+'.jpg', abu[y:y+h, x:x+w])
        # Menggambar kotak hijau di sekitar wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Menampilkan gambar dengan wajah yang terdeteksi
    cv2.imshow("Face Recognition Window", frame)
    # Mengambil foto sampai mencapai 100 pengambilan
    if (a > 100):
        break  # Keluar dari loop jika sudah mencapai 100 foto

# Melepas kamera dan menutup semua jendela OpenCV
video.release()
cv2.destroyAllWindows()

# Fungsi untuk mengambil gambar dan label (ID) dari folder Dataset
def getImagesWithLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))]  # Menyaring file gambar
    faceSamples = []  # Menyimpan gambar wajah
    Ids = []  # Menyimpan ID untuk masing-masing gambar wajah

    # Membaca setiap gambar di dalam folder
    for imagePath in imagePaths:
        try:
            # Membuka gambar dan mengonversi menjadi grayscale
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')  # Mengonversi gambar menjadi array numpy
            # Mendapatkan ID dari nama file gambar
            Id = int(os.path.split(imagePath)[-1].split(".")[1])  # ID diambil dari nama file
            # Mendeteksi wajah dalam gambar
            faces = detector.detectMultiScale(imageNp)  # Mendeteksi wajah dalam gambar

            for (x, y, w, h) in faces:
                # Menyimpan wajah yang terdeteksi
                faceSamples.append(imageNp[y:y+h, x:x+w])  # Menyimpan gambar wajah pada array
                Ids.append(Id)  # Menyimpan ID yang sesuai

        except Exception as e:
            # Menangani jika terjadi kesalahan pada pemrosesan gambar
            print(f"Error processing file {imagePath}: {e}")

    return faceSamples, Ids  # Mengembalikan wajah dan ID yang terdeteksi

# Memanggil fungsi untuk mendapatkan gambar dan ID dari folder Dataset
faces, Ids = getImagesWithLabels('Dataset')
recognizer.train(faces, np.array(Ids))  # Melakukan pelatihan model dengan data gambar wajah dan ID
# Menyimpan model pelatihan yang sudah dilatih ke dalam file training.xml
recognizer.save('Dataset/training.xml')
