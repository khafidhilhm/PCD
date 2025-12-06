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

# ==== REKOMENDASI TANAMAN ====
tanaman_cocok = {
    "Red soil": "🌾 Tanaman yang cocok: Kacang tanah, sorghum, millet.",
    "Black Soil": "🌱 Tanaman yang cocok: Kapas, jagung, gandum, kedelai.",
    "Alluvial soil": "🍃 Tanaman yang cocok: Padi, tebu, gandum, sayuran.",
    "Clay soil": "🌿 Tanaman yang cocok: Padi, brokoli, kubis, tanaman yang butuh banyak air."
}

# ==== PREDIKSI FUNCTION ====
def predict_soil(image_array):
    image_array = image_array.astype("float32")
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# ==== STREAMLIT UI ====
st.title("🌱 Sistem Klasifikasi Jenis Tanah")
st.write("Unggah gambar tanah untuk mengetahui jenis tanah dan rekomendasi tanaman yang cocok.")

uploaded_file = st.file_uploader("Pilih gambar tanah", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    img_np = np.array(img)
    img_np = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    img_np = img_np / 255.0
    img_np = np.expand_dims(img_np, axis=0)

    # Predict
    pred = predict_soil(img_np)[0]
    class_idx = np.argmax(pred)
    confidence = float(np.max(pred) * 100)

    soil_type = class_names[class_idx]

    st.subheader("📌 Hasil Prediksi")
    st.success(f"Jenis Tanah: **{soil_type}**")
    st.write(f"Tingkat Keyakinan: **{confidence:.2f}%**")

    # ======= REKOMENDASI TANAMAN BERDASARKAN JENIS TANAH =======
    if soil_type in tanaman_cocok:
        st.info(tanaman_cocok[soil_type])
    else:
        st.warning("Belum ada rekomendasi tanaman untuk jenis tanah ini.")
