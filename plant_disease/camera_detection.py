"""
FINAL CAMERA-BASED PLANT DISEASE DETECTION
✔ CNN Feature Extractor + SVM
✔ Severity %
✔ Fertilizer + Remedy
✔ Tamil Voice Output (Natural)
✔ Automatic Leaf Detection
✔ Frame Smoothing
"""

# =========================
# IMPORTS
# =========================
import cv2
import numpy as np
import tensorflow as tf
import json
import joblib
import os
import uuid
from collections import deque
from gtts import gTTS
from playsound import playsound
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# PATH CONFIG
# =========================
CNN_MODEL_PATH = "plant_disease_classifier.h5"
SVM_MODEL_PATH = "svm_classifier.pkl"
SCALER_PATH = "svm_scaler.pkl"
SEVERITY_MODEL_PATH = "severity_regressor.pkl"

AGRI_KNOWLEDGE_PATH = "agri_knowledge.json"
CLASS_INDICES_PATH = "class_indices.json"

IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.50
MIN_LEAF_AREA = 3000
SMOOTH_FRAMES = 3

# =========================
# LOAD MODELS
# =========================
print("🌿 Loading models...")

cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
svm = joblib.load(SVM_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

severity_model = None
try:
    severity_model = joblib.load(SEVERITY_MODEL_PATH)
    print("✅ Severity model loaded")
except:
    print("⚠ Severity model not found")

# Feature extractor layer
feature_extractor = tf.keras.Model(
    cnn_model.input,
    cnn_model.get_layer("feature_layer").output
)

print("✅ All models loaded successfully")

# =========================
# LOAD JSON FILES
# =========================
with open(AGRI_KNOWLEDGE_PATH, encoding="utf-8") as f:
    agri_knowledge = json.load(f)

with open(CLASS_INDICES_PATH, encoding="utf-8") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}

# =========================
# TAMIL VOICE FUNCTION
# =========================
def speak_tamil(text):
    try:
        filename = f"voice_{uuid.uuid4()}.mp3"
        tts = gTTS(text=text, lang="ta")
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        print("Voice Error:", e)

# =========================
# IMAGE PREPROCESS
# =========================
def preprocess(img):
    img = cv2.resize(img, IMG_SIZE)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# SEVERITY LABEL (Tamil)
# =========================
def severity_label(percent):
    if percent >= 80:
        return "அதிகம்"
    elif percent >= 60:
        return "மிதமான"
    else:
        return "குறைவான"

# =========================
# START CAMERA
# =========================
cap = cv2.VideoCapture(0)
pred_queue = deque(maxlen=SMOOTH_FRAMES)
last_spoken = ""

speak_tamil("செயற்கை நுண்ணறிவு தாவர நோய் கண்டறிதல் தொடங்கியது")

print("\n📷 Camera Started. Press 'q' to Quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # =========================
    # LEAF DETECTION USING HSV
    # =========================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area > MIN_LEAF_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            leaf = frame[y:y+h, x:x+w]

            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # =========================
            # PREDICTION
            # =========================
            img_input = preprocess(leaf)
            features = feature_extractor.predict(img_input, verbose=0)
            features_scaled = scaler.transform(features)

            probs = svm.predict_proba(features_scaled)[0]
            pred_queue.append(probs)

            if len(pred_queue) == SMOOTH_FRAMES:
                avg_probs = np.mean(pred_queue, axis=0)
                class_id = int(np.argmax(avg_probs))
                confidence = float(avg_probs[class_id])

                if confidence >= CONF_THRESHOLD:
                    disease = index_to_class[class_id]
                    remedy = agri_knowledge[disease]["remedy"]
                    fertilizer = agri_knowledge[disease]["fertilizer"]

                    # Severity %
                    sev_percent = confidence * 100
                    if severity_model:
                        sev_percent = float(
                            severity_model.predict(features)[0]
                        )

                    sev_label = severity_label(sev_percent)

                    label = f"{disease.replace('___',' - ')} | {round(sev_percent,1)}%"
                    cv2.putText(display, label, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                    if disease != last_spoken:
                        print("\n🌿 Disease:", disease)
                        print("Severity:", sev_label, f"({round(sev_percent,1)}%)")
                        print("Remedy:", remedy)
                        print("Fertilizer:", fertilizer)

                        # =========================
                        # TAMIL VOICE OUTPUT
                        # =========================
                        voice_text = f"""
                        இந்த இலைக்கு {disease.replace('___',' ')} நோய் கண்டறியப்பட்டுள்ளது.
                        நோயின் தீவிரம் {sev_label}.
                        சிகிச்சை: {remedy}.
                        பரிந்துரைக்கப்படும் உரம்: {fertilizer}.
                        """

                        speak_tamil(voice_text)

                        last_spoken = disease

    cv2.imshow("Smart Plant Disease Detection", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()

speak_tamil("நோய் கண்டறிதல் நிறுத்தப்பட்டது")
print("👋 Program Closed")
