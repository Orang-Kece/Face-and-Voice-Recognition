# Mengimpor library yang dibutuhkan
import os  # Digunakan untuk mengelola operasi file dan direktori
from pydub import AudioSegment  # Digunakan untuk memanipulasi file audio
import librosa  # Digunakan untuk analisis audio (meskipun tidak digunakan langsung di sini)
import soundfile as sf  # Digunakan untuk menyimpan file audio dalam format lain (juga tidak digunakan di sini)

# Konfigurasi
input_dir = r"D:\Kuliah\Sem 7\Citra\After UTS\UAS\ML-face-recognition-main\Voice\dataset"  # Direktori yang berisi file audio mentah
output_dir = r"D:\Kuliah\Sem 7\Citra\After UTS\UAS\ML-face-recognition-main\Voice\Processed"  # Direktori untuk menyimpan dataset yang sudah diproses
clip_duration = 5 * 1000  # Durasi tiap klip dalam milidetik (misalnya, 5 detik), dikalikan 1000 untuk mendapatkan milidetik
sample_rate = 16000  # Target sample rate untuk file audio yang diproses (16kHz)
step_size = clip_duration  # Ukuran langkah untuk jendela geser (tidak ada overlap, jika menggunakan clip_duration // 2 berarti overlap 50%)

# Memastikan direktori output ada
os.makedirs(output_dir, exist_ok=True)  # Membuat direktori output jika belum ada, tidak menimbulkan error jika sudah ada

def process_audio(file_path, speaker_name, output_dir, clip_duration, sample_rate, step_size, clip_counter):
    """
    Memproses sebuah file audio: meresample, membagi menjadi klip-klip, dan menyimpannya.
    """
    try:
        # Memuat file audio menggunakan pydub
        audio = AudioSegment.from_file(file_path)  # Membaca file audio
        audio = audio.set_frame_rate(sample_rate)  # Meresample audio ke sample rate yang ditentukan
        audio = audio.set_channels(1)  # Mengonversi audio menjadi mono (1 channel) jika diperlukan

        # Membuat direktori khusus untuk speaker
        speaker_dir = os.path.join(output_dir, speaker_name)  # Membuat direktori untuk speaker
        os.makedirs(speaker_dir, exist_ok=True)  # Membuat direktori speaker jika belum ada

        # Mengambil nama file asli tanpa ekstensi
        file_base = os.path.splitext(os.path.basename(file_path))[0]

        # Membagi audio menjadi klip-klip
        for i in range(0, len(audio), step_size):
            clip = audio[i:i + clip_duration]  # Memotong audio menjadi klip dengan durasi tertentu
            if len(clip) > 0:  # Memasukkan klip terakhir meskipun lebih pendek dari durasi yang diinginkan
                # Menyusun nama file untuk klip
                clip_name = f"{file_base}_clip{clip_counter}.wav"
                clip_path = os.path.join(speaker_dir, clip_name)  # Path untuk menyimpan klip
                clip.export(clip_path, format="wav")  # Menyimpan klip dalam format WAV
                clip_counter += 1  # Increment counter untuk nama klip berikutnya

        print(f"Processed {clip_counter} total clips for speaker '{speaker_name}' so far from file: {file_path}")
        return clip_counter  # Mengembalikan jumlah klip yang telah diproses
    except Exception as e:
        print(f"Error processing {file_path}: {e}")  # Menangani jika ada error
        return clip_counter  # Mengembalikan counter meskipun terjadi error

def process_dataset(input_dir, output_dir, clip_duration, sample_rate, step_size):
    """
    Memproses seluruh dataset: iterasi untuk setiap speaker dan file audio mereka.
    """
    for speaker in os.listdir(input_dir):  # Iterasi untuk setiap folder speaker dalam direktori input
        speaker_dir = os.path.join(input_dir, speaker)  # Mendapatkan path untuk folder speaker
        if os.path.isdir(speaker_dir):  # Memastikan hanya folder yang diproses
            print(f"Processing files for speaker: {speaker}")
            clip_counter = 0  # Inisialisasi counter untuk setiap speaker
            for file in os.listdir(speaker_dir):  # Iterasi untuk setiap file dalam folder speaker
                file_path = os.path.join(speaker_dir, file)  # Mendapatkan path untuk file audio
                if file.endswith(".wav"):  # Memastikan hanya file WAV yang diproses
                    clip_counter = process_audio(file_path, speaker, output_dir, clip_duration, sample_rate, step_size, clip_counter)

# Menjalankan pemrosesan dataset
process_dataset(input_dir, output_dir, clip_duration, sample_rate, step_size)

print("Audio preprocessing complete!")  # Menampilkan pesan jika pemrosesan selesai
