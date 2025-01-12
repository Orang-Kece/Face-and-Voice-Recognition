# Mengimpor library yang dibutuhkan
import os  # Digunakan untuk berinteraksi dengan sistem file (misalnya membaca direktori)
import numpy as np  # Digunakan untuk operasi numerik dengan array
from sklearn.mixture import GaussianMixture  # Digunakan untuk membuat model Gaussian Mixture Model (GMM)
from python_speech_features import mfcc  # Digunakan untuk ekstraksi fitur MFCC dari sinyal audio
from scipy.io import wavfile  # Digunakan untuk membaca dan menulis file WAV
import joblib  # Digunakan untuk menyimpan model menggunakan format serialisasi joblib
from scipy.io.wavfile import write  # Digunakan untuk menulis file WAV
import librosa  # Digunakan untuk memproses audio, seperti ekstraksi MFCC

def extract_features(audio_path, n_mfcc=13, target_sample_rate=16000):
    """
    Fungsi untuk mengekstraksi fitur MFCC dari file audio menggunakan librosa.
    Menggunakan sample rate target untuk mengubah sampling rate file audio.
    """
    # Memuat file audio dan mengubah sample rate-nya
    signal, sample_rate = librosa.load(audio_path, sr=target_sample_rate)
    # Menghitung MFCC dari sinyal audio
    features = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc).T
    return features

def split_audio(signal, sample_rate, segment_duration=5):
    """
    Fungsi untuk membagi audio menjadi segmen-segmen berdurasi tetap (default 5 detik).
    """
    segment_length = segment_duration * sample_rate  # Menghitung panjang setiap segmen dalam sampel
    return [signal[i:i + segment_length] for i in range(0, len(signal), segment_length // 2)
            if len(signal[i:i + segment_length]) == segment_length]  # Membagi sinyal ke dalam segmen-segmen

def extract_features(audio_path, n_mfcc=13):
    """
    Fungsi untuk mengekstraksi fitur MFCC dari file audio menggunakan python_speech_features.
    """
    # Membaca file audio dengan scipy
    sample_rate, signal = wavfile.read(audio_path)
    # Mengekstraksi MFCC dari sinyal audio
    features = mfcc(signal, samplerate=sample_rate, numcep=n_mfcc)
    return features

def train_gmm(features, n_components=16, max_iter=200):
    """
    Fungsi untuk melatih Gaussian Mixture Model (GMM) dengan fitur yang telah diekstraksi.
    """
    # Membuat objek GMM dan melatihnya dengan fitur yang diberikan
    gmm = GaussianMixture(n_components=n_components, max_iter=max_iter, covariance_type='diag', n_init=3)
    gmm.fit(features)
    return gmm

def main(audio_directory, model_output_dir):
    """
    Fungsi utama untuk melatih GMM untuk identifikasi pembicara berdasarkan dataset audio.
    """
    # Memastikan direktori model output ada
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    # Mendapatkan semua nama pembicara (direktori di dalam audio_directory)
    speakers = [d for d in os.listdir(audio_directory) if os.path.isdir(os.path.join(audio_directory, d))]

    # Proses untuk setiap pembicara dalam dataset
    for speaker in speakers:
        speaker_path = os.path.join(audio_directory, speaker)
        print(f"Processing speaker: {speaker}")

        # Mengambil fitur dari semua file WAV untuk pembicara ini
        features = np.vstack([extract_features(os.path.join(speaker_path, f))
                              for f in os.listdir(speaker_path) if f.endswith('.wav')])

        # Melatih GMM untuk pembicara ini
        print(f"Training GMM for speaker: {speaker}")
        gmm = train_gmm(features)

        # Menyimpan model GMM untuk pembicara ini
        model_path = os.path.join(model_output_dir, f"{speaker}.gmm")
        print(f"Saving model to {model_path}")
        joblib.dump(gmm, model_path)  # Menyimpan model menggunakan joblib

if __name__ == "__main__":
    # Menetapkan direktori dataset dan direktori output untuk model
    audio_directory = r"D:\Kuliah\Sem 7\Citra\After UTS\UAS\ML-face-recognition-main\Voice\Processed"  # Path ke folder dataset audio
    model_output_dir = r"D:\Kuliah\Sem 7\Citra\After UTS\UAS\ML-face-recognition-main\Voice\gmm"  # Path untuk menyimpan model GMM

    # Menjalankan fungsi utama
    main(audio_directory, model_output_dir)
