import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model

# Load model and class labels
model = load_model('best_asl_model.h5')
with open("class_labels.pkl", "rb") as f:
    class_labels = pickle.load(f)

IMG_HEIGHT, IMG_WIDTH = 64, 64

# GUI setup
root = tk.Tk()
root.title("ASL Sign Recognition")
root.geometry("500x600")
root.configure(bg="#f0f0f0")

# Display area
img_label = Label(root)
img_label.pack(pady=10)

result_label = Label(root, text="", font=("Helvetica", 16), bg="#f0f0f0")
result_label.pack(pady=10)

# 🧠 Prediction function
def predict_image(img_path):
    img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)
    prediction = model.predict(img_array)[0]
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    result_label.config(text=f"Prediction: {predicted_class} ({confidence:.2f})")

# 📁 Upload image
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk
        predict_image(file_path)

# 🎥 Webcam prediction
def webcam_predict():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        frame_array = frame_resized / 255.0
        frame_array = frame_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)
        prediction = model.predict(frame_array)[0]
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        cv2.putText(frame, f"{predicted_class} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("ASL Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Buttons
Button(root, text="Upload Image", command=upload_image, font=("Helvetica", 14), bg="#4CAF50", fg="white").pack(pady=10)
Button(root, text="Use Webcam", command=webcam_predict, font=("Helvetica", 14), bg="#2196F3", fg="white").pack(pady=10)

root.mainloop()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info/warnings is main s webcam wala code khtm kr do main just upload s predicton gi give me refained and beautiful code