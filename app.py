import tkinter as tk
import subprocess
import threading
import time

# Flags to control the loops
stop_scan = threading.Event()
stop_detecteb = threading.Event()

# Subprocess references for termination
scan_process = None
detecteb_process = None

def run_scan_loop():
    """
    Run the 'scan.py' script in a loop until stop_scan is set.
    """
    global scan_process
    try:
        result_label.config(text="Running scan.py in a loop...", fg="black")
        while not stop_scan.is_set():
            # Start scan.py process
            scan_process = subprocess.Popen(
                ["python", "scan.py"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            scan_process.wait()  # Wait for the process to finish
            time.sleep(1)  # Delay between iterations
            if stop_scan.is_set():
                break
    except Exception as e:
        result_label.config(text=f"Error in scan.py: {e}", fg="red")
    finally:
        if scan_process:
            scan_process.terminate()
        result_label.config(text="scan.py loop stopped.", fg="blue")

def run_detecteb_loop():
    """
    Run the 'detecteb.py' script in a loop until stop_detecteb is set.
    """
    global detecteb_process
    try:
        result_label.config(text="Running detecteb.py in a loop...", fg="black")
        while not stop_detecteb.is_set():
            # Show "Recording audio" status
            result_label.config(text="Recording audio...", fg="orange")
            
            # Start detecteb.py process
            detecteb_process = subprocess.Popen(
                ["python", "detecteb.py"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            stdout, _ = detecteb_process.communicate()  # Get the output
            last_line = stdout.splitlines()[-1] if stdout else "No output"
            
            # Update with the actual result
            result_label.config(text=f"detecteb.py: {last_line}", fg="green")
            
            time.sleep(1)  # Delay before the next iteration
            if stop_detecteb.is_set():
                break
    except Exception as e:
        result_label.config(text=f"Error in detecteb.py: {e}", fg="red")
    finally:
        if detecteb_process:
            detecteb_process.terminate()
        result_label.config(text="detecteb.py loop stopped.", fg="blue")

def start_processes():
    """
    Start both scripts in separate threads and set the stop flags to False.
    """
    start_button.config(state="disabled")
    stop_scan.clear()
    stop_detecteb.clear()
    result_label.config(text="Starting both loops...", fg="black")
    
    scan_thread = threading.Thread(target=run_scan_loop, daemon=True)
    detecteb_thread = threading.Thread(target=run_detecteb_loop, daemon=True)
    
    scan_thread.start()
    detecteb_thread.start()

def stop_processes():
    """
    Stop the loops and terminate any running subprocesses.
    """
    stop_scan.set()
    stop_detecteb.set()
    if scan_process:
        scan_process.terminate()
    if detecteb_process:
        detecteb_process.terminate()
    start_button.config(state="normal")
    result_label.config(text="Processes stopped.", fg="blue")

# Create the main application window
app = tk.Tk()
app.title("Face & Voice Recognition")
app.geometry("400x300")

title_label = tk.Label(app, text="Face & Voice Recognition", font=("Arial", 18, "bold"), pady=20)
title_label.pack()

start_button = tk.Button(app, text="Start", command=start_processes, width=20, height=2)
start_button.pack(pady=10)

stop_button = tk.Button(app, text="Stop", command=stop_processes, width=20, height=2)
stop_button.pack(pady=10)

result_label = tk.Label(app, text="Output will appear here.", width=40, height=4)
result_label.pack(pady=20)

app.mainloop()
