
#   **Trigger Word Detection Model**  

This repository contains an AI-powered **Trigger Word Detection** system that listens for a specific keyword ("robo") in an audio stream. The model processes audio in **real-time** and can be integrated into voice assistants, smart devices, or hands-free control systems.  

## **📌 Project Overview**  
🔹 Implemented trigger word detection from scratch using **deep learning**  
🔹 Built a **custom speech dataset** with positive (trigger word) and negative samples  
🔹 Used **MFCC features & spectrograms** for audio preprocessing  
🔹 **Real-time detection** without saving to `.wav` files  
🔹 **Full implementation in a step-by-step Jupyter Notebook**  

## **🚀 Installation & Setup**  
### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/Moaz0009/Trigger-Word-Detection-Model.git
cd Trigger-Word-Detection-Model
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Run Real-Time Trigger Word Detection**  
```bash
python main.py
```

## **🛠️ How It Works**  
### **🔹 Step 1: Data Synthesis**  
We generate a **custom dataset** by mixing **trigger words**, **random words**, and **background noise**.  
```python
from pydub import AudioSegment
import numpy as np

# Load background noise & speech samples
background = AudioSegment.from_wav("data_set/bk/chunk_024.wav")
speech = AudioSegment.from_wav("data_set/dd/detect (13).wav")
```

### **🔹 Step 2: Feature Extraction (MFCC & Spectrograms)**  
Convert raw audio into features suitable for deep learning.  
```python
from trigger_util import get_mfcc_features

features = get_mfcc_features("data_set/dd/detect (13).wav")
```

### **🔹 Step 3: Model Training**  
We use a **deep learning model (LSTM/CNN-based)** to detect the trigger word.  
```python
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense

model = Sequential([
    Conv1D(filters=64, kernel_size=5, activation="relu"),
    LSTM(units=128, return_sequences=True),
    Dense(units=1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

### **🔹 Step 4: Real-Time Inference**  
Instead of processing `.wav` files, the model directly processes **raw audio arrays** for **low-latency detection**.  
```python
def detect_trigger(audio_array):
    features = extract_features(audio_array)
    prediction = model.predict(features)
    return prediction > 0.5
```


## **🤝 Contributing**  
Contributions are welcome! Feel free to open issues or pull requests.  

## **📜 License**  
This project is licensed under the **MIT License**.  
