# 🖐 ASL Sign Language Recognition

A deep learning project that recognizes **American Sign Language (ASL) hand signs** using a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**.  
The project includes:
- **Dataset preprocessing & augmentation**
- **Model training with class balancing**
- **Image upload-based prediction using Tkinter GUI**

---

## 📌 Features
✅ Data preprocessing with augmentation (rotation, zoom, shift)  
✅ CNN model trained with class balancing  
✅ Save and load trained model & labels for inference  
✅ Tkinter GUI for uploading images and getting predictions  
✅ Shows **predicted ASL letter** with **confidence score**

---

## 📂 Project Structure
```bash
ASL_Project/
│── dataset/ # ASL dataset (training images)
│── asl_train.py # Data preprocessing + model training script
│── asl_gui.py # Tkinter GUI for predictions
│── best_asl_model.h5 # Trained CNN model
│── class_labels.pkl # Saved class labels
│── README.md # Project documentation
```
---


## 📊 Dataset
We use the **ASL Alphabet Dataset**, which contains folders for each ASL sign (letters A–Z, plus additional classes like "space", "nothing", etc.).

**Dataset Structure:**


dataset/
│── A/
│── B/
│── C/
│── ...
- **Training Images**: High-resolution images of hands forming each ASL sign.
- **Class Count**: Matches the number of folders in the dataset.
- **Important Note**:  
  - **Do NOT** apply horizontal flipping for ASL gestures — flipping can change the meaning of the sign.

---

## 🔄 Workflow

### **1️⃣ Data Preprocessing**
- Rescale images to `1./255` for normalization.
- Resize all images to **64×64 pixels** for consistent input shape.
- Apply data augmentation:
  - Rotation (`±20°`)
  - Zoom (up to 20%)
  - Width/Height shift (up to 10%)
  - Shearing (15%)
- Split dataset into:
  - **80% Training**
  - **20% Validation**

### **2️⃣ Class Label Preparation**
- Extract all folder names as **class labels**.
- Save labels into `class_labels.pkl` for use during prediction.

### **3️⃣ Model Training**
- CNN architecture:
  - 3 convolutional layers with Batch Normalization
  - MaxPooling after each conv block
  - Dropout for regularization
  - Fully connected Dense layer before output
- Loss Function: **Categorical Crossentropy**
- Optimizer: **Adam**
- Metrics: **Accuracy**
- Training Strategy:
  - Early Stopping (patience=2)
  - Reduce Learning Rate on Plateau
  - Save best model as `best_asl_model.h5`

### **4️⃣ Prediction Phase**
- Load `best_asl_model.h5` and `class_labels.pkl`.
- Accept uploaded image from GUI.
- Preprocess image:
  - Resize to **64×64**
  - Normalize pixels (`1./255`)
- Model outputs:
  - **Predicted Class**
  - **Confidence Score**








