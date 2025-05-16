import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import cv2
import time
from video_monitor import monitor_video

shared_frame = None
monitoring_running = False

# Example questions
QUESTIONS = [
    {
        "question": "What is the capital of France?",
        "options": ["London", "Berlin", "Paris", "Madrid"],
        "answer": 2
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "options": ["Venus", "Mars", "Jupiter", "Saturn"],
        "answer": 1
    },
    {
        "question": "What is 2 + 2?",
        "options": ["3", "4", "5", "6"],
        "answer": 1
    }
]

def frame_callback(frame):
    global shared_frame
    shared_frame = frame
    print("Frame received in callback:", type(frame), frame.shape if frame is not None else None)

def log_callback(msg):
    print(msg)

def start_detection():
    monitor_video(log_callback, frame_callback)

class MonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Exam Monitoring System")
        self.root.geometry("1000x600")
        self.started = False

        # Timer
        self.exam_duration = 60 * 60  # 1 hour in seconds
        self.time_left = self.exam_duration
        self.timer_label = tk.Label(root, text="Time Remaining: 01:00:00", font=("Arial", 16), fg="red")
        self.timer_label.pack(pady=10, anchor="n")

        # Main frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill="both", expand=True)

        # Left: Questions
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        self.question_index = 0
        self.selected_option = tk.IntVar()
        self.question_label = tk.Label(self.left_frame, text="Click 'Start Test' to begin.", font=("Arial", 14), wraplength=400, justify="left")
        self.question_label.pack(anchor="w", pady=(0, 10))
        self.options_frame = tk.Frame(self.left_frame)
        self.options_frame.pack(anchor="w")

        self.next_button = tk.Button(self.left_frame, text="Next", command=self.next_question, state="disabled")
        self.next_button.pack(anchor="e", pady=10)

        # Right: Camera feed
        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.pack(side="right", fill="y", padx=20, pady=20)

        self.camera_label = tk.Label(self.right_frame, text="Camera Feed", font=("Arial", 12))
        self.camera_label.grid(row=0, column=0, pady=(0, 5))

        self.image_label = tk.Label(self.right_frame, width=120, height=90, bg="black")
        self.image_label.grid(row=1, column=0, pady=(0, 10))

        self.start_button = tk.Button(self.right_frame, text="Start Test", command=self.start_test, width=20, height=2)
        self.start_button.grid(row=2, column=0, pady=(0, 10))

        self.status = tk.Label(self.right_frame, text="", fg="green")
        self.status.grid(row=3, column=0, pady=(0, 10))

        self.right_frame.grid_rowconfigure(1, minsize=240)
        self.right_frame.grid_columnconfigure(0, minsize=320)

        self.update_camera()
        self.update_timer()

    def start_test(self):
        global monitoring_running
        if self.started:
            messagebox.showinfo("Info", "Test already running.")
            return

        self.started = True
        self.status.config(text="Monitoring started...")
        monitoring_running = True
        threading.Thread(target=start_detection, daemon=True).start()
        self.time_left = self.exam_duration  # Reset timer
        self.update_timer()
        self.show_question()
        self.next_button.config(state="normal")

    def update_camera(self):
        global shared_frame
        if self.started and shared_frame is not None:
            print("Displaying frame in GUI")
            frame_rgb = cv2.cvtColor(shared_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img.resize((160, 120)))
            self.image_label.imgtk = imgtk
            self.image_label.configure(image=imgtk)
        self.root.after(30, self.update_camera)  # ~30 FPS

    def update_timer(self):
        if self.started and self.time_left > 0:
            self.time_left -= 1
            mins, secs = divmod(self.time_left, 60)
            hours, mins = divmod(mins, 60)
            self.timer_label.config(text=f"Time Remaining: {hours:02d}:{mins:02d}:{secs:02d}")
            self.root.after(1000, self.update_timer)
        elif self.started and self.time_left == 0:
            self.timer_label.config(text="Time's up!")
            messagebox.showinfo("Time's up!", "The exam time has ended.")
            self.started = False

    def show_question(self):
        q = QUESTIONS[self.question_index]
        self.question_label.config(text=f"Q{self.question_index+1}: {q['question']}")
        for widget in self.options_frame.winfo_children():
            widget.destroy()
        self.selected_option.set(-1)
        for idx, opt in enumerate(q['options']):
            rb = tk.Radiobutton(self.options_frame, text=opt, variable=self.selected_option, value=idx, font=("Arial", 12))
            rb.pack(anchor="w")

    def next_question(self):
        if self.selected_option.get() == -1:
            messagebox.showwarning("Warning", "Please select an option before proceeding.")
            return
        if self.question_index < len(QUESTIONS) - 1:
            self.question_index += 1
            self.show_question()
        else:
            messagebox.showinfo("Exam Finished", "You have completed the exam!")
            self.started = False
            self.next_button.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = MonitorApp(root)
    root.mainloop()
