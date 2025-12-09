import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import json

# ==== LOAD MODEL ====
interpreter = tf.lite.Interpreter(model_path="soil_model_quantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 220

# ==== LOAD CLASS NAMES ====
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ==== PREDIKSI FUNCTION ====
def predict_soil(image_array):
    image_array = image_array.astype("float32")
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# ==== STREAMLIT UI ====
st.title("Klasifikasi Tanah Menggunakan Model TFLite")
st.write("Unggah gambar tanah untuk mengklasifikasikan jenisnya.")

uploaded_file = st.file_uploader("Pilih gambar tanah", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang Diunggah", use_container_width=True)

    img_np = np.array(img)
    img_np = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    img_np = img_np / 255.0
    img_np = np.expand_dims(img_np, axis=0)

    # Predict
    pred = predict_soil(img_np)[0]
    class_idx = np.argmax(pred)
    confidence = float(np.max(pred) * 100)

    st.subheader("Hasil Prediksi")
    st.success(f"Jenis Tanah yang Diprediksi: **{class_names[class_idx]}**")
    st.write(f"Tingkat keakuratan: **{confidence:.2f}%**")
