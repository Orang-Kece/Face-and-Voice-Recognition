# Mengimpor library yang dibutuhkan
import tkinter as tk  # Digunakan untuk membuat antarmuka pengguna grafis (GUI)
import subprocess  # Digunakan untuk menjalankan skrip eksternal dari Python

def run_scripts():
    """
    Fungsi untuk menjalankan kedua skrip scan.py dan detecteb.py.
    """
    try:
        # Menjalankan skrip scan.py dan menangkap outputnya
        subprocess.run(["python", "scan.py"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Memperbarui UI untuk menunjukkan "Recording voice" sebelum menjalankan detecteb.py
        result_label.config(text="Recording voice...")

        # Menjalankan skrip detecteb.py dan menangkap outputnya
        result = subprocess.run(["python", "detecteb.py"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Mengambil baris terakhir dari output detecteb.py
        last_line = result.stdout.splitlines()[-1]  # Mengambil baris terakhir yang dicetak di terminal

        # Memperbarui UI dengan hasil dari detecteb.py setelah perekaman selesai
        result_label.config(text=last_line)
    except Exception as e:
        # Menangani jika terjadi kesalahan saat menjalankan skrip
        print(f"Error running scripts: {e}")
        result_label.config(text=f"Error: {e}")  # Memperbarui UI dengan pesan kesalahan

# Membuat jendela utama untuk aplikasi GUI
app = tk.Tk()
app.title("Script Runner")  # Menetapkan judul jendela
app.geometry("400x200")  # Menetapkan ukuran jendela aplikasi

# Membuat tombol Start untuk memulai proses menjalankan skrip
start_button = tk.Button(app, text="Start", command=run_scripts, width=20, height=2)
start_button.pack(pady=30)  # Menempatkan tombol dengan padding vertikal

# Membuat label untuk menampilkan output dari skrip detecteb.py
result_label = tk.Label(app, text="Output will appear here.", width=40, height=4)
result_label.pack(pady=10)  # Menempatkan label dengan padding vertikal

# Menjalankan aplikasi GUI
app.mainloop()  # Menjalankan aplikasi tkinter untuk menampilkan antarmuka pengguna
