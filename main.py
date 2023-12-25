import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from deepface import DeepFace

class EmotionDetectionApp:
    def __init__(self, root):
        # Initialize the GUI
        self.root = root
        self.root.title("Emotion Detection App")
        self.setup_ui()

        # Initialize variables
        self.cap = None
        self.is_capturing = False
        self.model = DeepFace.build_model("Emotion")
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def setup_ui(self):
        # Color scheme
        bg_color = "#2a446a"
        label_color = "#f79633"
        button_color = "#ffcc00"
        button_text_color = "#2a446a"

        # Create style
        style = ttk.Style()
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=label_color)
        style.configure("TButton", background=button_color, foreground=button_text_color, font=("Helvetica", 12))

        # Create main frame
        self.main_frame = ttk.Frame(self.root, style="TFrame")
        self.main_frame.pack(expand=True, fill="both")

        # Create label for camera feed
        self.label = ttk.Label(self.main_frame, style="TLabel")
        self.label.pack(expand=True, fill="both")

        # Create label for displaying emotion
        self.emotion_label = ttk.Label(self.main_frame, text="", font=("Helvetica", 16), style="TLabel")
        self.emotion_label.pack(pady=(0, 20))

        # Create buttons frame
        buttons_frame = ttk.Frame(self.root, style="TFrame")
        buttons_frame.pack(expand=True, fill="both")

        # Create Start and Exit buttons
        start_button = ttk.Button(buttons_frame, text="Start", command=self.start_camera, style='TButton', cursor='hand2')
        start_button.pack(side="left", padx=(10, 5), pady=10, expand=True, fill="both")

        exit_button = ttk.Button(buttons_frame, text="Exit", command=self.exit_app, style='TButton', cursor='hand2')
        exit_button.pack(side="right", padx=(5, 10), pady=10, expand=True, fill="both")

    def start_camera(self):
        # Start capturing video feed from the camera
        if not self.is_capturing:
            self.cap = cv2.VideoCapture(0)

            if self.cap.isOpened():
                self.is_capturing = True
                self.show_camera()
            else:
                print("Error: Unable to open the camera.")

    def show_camera(self):
        # Continuously capture and display frames from the camera
        if self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                emotion = self.detect_emotion(frame)
                self.display_frame(frame, emotion)
                self.root.after(10, self.show_camera)
            else:
                print("Error: Unable to read a frame.")
                self.is_capturing = False
                self.cap.release()

    def detect_emotion(self, frame):
        # Detect emotion in the given frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(gray_frame)

        if faces is not None and len(faces) > 0:
            face = faces[0]
            face_roi = gray_frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
            resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
            normalized_face = resized_face / 255.0
            reshaped_face = normalized_face.reshape(1, 48, 48, 1)
            preds = self.model.predict(reshaped_face)[0]
            emotion_idx = preds.argmax()
            emotion = self.emotion_labels[emotion_idx]
        else:
            emotion = "No face detected"

        return emotion

    def detect_faces(self, gray_frame):
        # Detect faces in the given grayscale frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def display_frame(self, frame, emotion):
        # Display the processed frame and update the emotion label
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        self.draw_bounding_box(image)
        self.draw_facial_landmarks(image)
        photo = ImageTk.PhotoImage(image=image)
        self.label.config(image=photo)
        self.label.image = photo

        self.emotion_label.config(text=f"Emotion: {emotion}")

    def draw_bounding_box(self, image):
        # Draw a bounding box around the detected face
        draw = ImageDraw.Draw(image)
        faces = self.detect_faces(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY))
        for (x, y, w, h) in faces:
            draw.rectangle([x, y, x+w, y+h], outline="#ffcc00", width=2)

    def draw_facial_landmarks(self, image):
        # Draw facial landmarks (dummy function, replace with actual logic)
        draw = ImageDraw.Draw(image)
        faces = self.detect_faces(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY))

        if faces is not None and len(faces) > 0:
            x, y, w, h = faces[0]
            landmarks = [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]
            for point in landmarks:
                draw.point(point, fill="#ffcc00")

    def exit_app(self):
        # Stop capturing, release resources, and close the application
        if self.is_capturing:
            self.is_capturing = False
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()
