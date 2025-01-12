# Mengimpor library yang dibutuhkan
import os  # Untuk berinteraksi dengan sistem file
import numpy as np  # Untuk operasi numerik dengan array
import joblib  # Untuk memuat dan menyimpan model yang telah dilatih
from python_speech_features import mfcc  # Untuk mengekstraksi fitur MFCC dari audio
from scipy.io import wavfile  # Untuk membaca file WAV
import sounddevice as sd  # Untuk merekam audio dari mikrofon
import wavio  # Untuk menyimpan file audio dalam format WAV

def record_audio(output_file, duration=5, sample_rate=16000):
    """
    Fungsi untuk merekam audio dari mikrofon dan menyimpannya ke file.
    """
    print("Recording... Speak into the microphone.")
    # Merekam audio dengan durasi yang ditentukan
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Menunggu hingga proses rekaman selesai
    # Menyimpan hasil rekaman ke file WAV
    wavio.write(output_file, audio, sample_rate, sampwidth=2)
    print(f"Recording saved to {output_file}")

def extract_features(audio_path, n_mfcc=13):
    """
    Fungsi untuk mengekstraksi fitur MFCC dari file audio.
    """
    # Membaca file audio menggunakan scipy
    sample_rate, signal = wavfile.read(audio_path)
    # Mengekstraksi fitur MFCC dari sinyal audio
    features = mfcc(signal, samplerate=sample_rate, numcep=n_mfcc)
    return features

def load_gmm_models(model_dir):
    """
    Fungsi untuk memuat semua model GMM dari direktori model.
    """
    gmm_models = {}
    # Mengiterasi setiap file di direktori model
    for model_file in os.listdir(model_dir):
        if model_file.endswith('.gmm'):
            speaker_name = model_file.split('.')[0]  # Menentukan nama pembicara dari nama file
            model_path = os.path.join(model_dir, model_file)
            # Memuat model GMM menggunakan joblib
            gmm = joblib.load(model_path)
            gmm_models[speaker_name] = gmm
    return gmm_models

def predict_speaker(features, gmm_models):
    """
    Fungsi untuk memprediksi pembicara berdasarkan fitur yang diekstraksi dan model GMM.
    """
    best_speaker = None
    best_score = -np.inf  # Memulai dengan skor terendah
    # Menghitung skor untuk setiap model GMM
    for speaker, gmm in gmm_models.items():
        score = gmm.score(features)
        if score > best_score:  # Jika skor lebih baik, perbarui pembicara terbaik
            best_score = score
            best_speaker = speaker
    return best_speaker

def main(model_dir, recording_duration=5):
    """
    Fungsi utama untuk merekam audio, menyimpannya ke file, dan mengidentifikasi pembicara.
    """
    # Path untuk menyimpan rekaman audio
    audio_file = "recorded_audio.wav"
    
    # Merekam audio dari mikrofon
    record_audio(audio_file, duration=recording_duration)
    
    # Mengekstraksi fitur dari audio yang direkam
    print(f"Extracting features from {audio_file}")
    features = extract_features(audio_file)
    
    # Memuat model GMM yang telah dilatih
    print(f"Loading GMM models from {model_dir}")
    gmm_models = load_gmm_models(model_dir)
    
    # Memprediksi pembicara
    print("Predicting the speaker...")
    speaker = predict_speaker(features, gmm_models)
    
    print(f"The speaker is: {speaker}")

if __name__ == "__main__":
    # Ganti path ini dengan path direktori model Anda yang sesungguhnya
    model_dir = r"D:\Kuliah\Sem 7\Citra\After UTS\UAS\ML-face-recognition-main\Voice\gmm"  # Path ke folder model GMM yang telah dilatih

    # Menjalankan fungsi utama
    main(model_dir, recording_duration=5)
