import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import json

# =======================
#  CUSTOM PAGE CONFIG
# =======================
st.set_page_config(
    page_title="Soil Classification",
    page_icon="🌱",
    layout="centered",
)

# =======================
#  LOAD MODEL
# =======================
interpreter = tf.lite.Interpreter(model_path="soil_model_quantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 220

# =======================
#  LOAD CLASS NAMES
# =======================
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# =======================
#  REKOMENDASI TANAMAN
# =======================
tanaman_cocok = {
    "Red soil": {
        "emoji": "🧱",
        "tanaman": "Kacang tanah, sorghum, millet.",
        "deskripsi": "Tanah berwarna merah karena kandungan besi, cocok untuk tanaman yang tahan kondisi kering."
    },
    "Black Soil": {
        "emoji": "🖤",
        "tanaman": "Kapas, jagung, gandum, kedelai.",
        "deskripsi": "Tanah hitam subur dengan kandungan mineral tinggi, mampu menahan air dengan baik."
    },
    "Alluvial soil": {
        "emoji": "🏞️",
        "tanaman": "Padi, tebu, gandum, sayuran.",
        "deskripsi": "Tanah endapan sungai yang sangat subur, baik untuk hampir semua jenis tanaman."
    },
    "Clay soil": {
        "emoji": "🟫",
        "tanaman": "Padi, brokoli, kubis, dan tanaman yang butuh air banyak.",
        "deskripsi": "Tanah liat yang padat dan mampu menahan air, cocok untuk tanaman dengan kebutuhan air tinggi."
    }
}

# =======================
#  PREDIKSI FUNCTION
# =======================
def predict_soil(image_array):
    image_array = image_array.astype("float32")
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output


# =======================
#  UI HEADER
# =======================
st.markdown("""
<h1 style='text-align:center; color:#2E7D32;'>
🌱 Sistem Klasifikasi Jenis Tanah
</h1>
<p style='text-align:center; font-size:17px;'>
Unggah foto tanah untuk mengetahui jenis tanah dan rekomendasi tanaman yang sesuai.
</p>
""", unsafe_allow_html=True)


# =======================
#  FILE UPLOADER
# =======================
uploaded_file = st.file_uploader("📸 Unggah gambar tanah", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    st.markdown("### 🖼️ Gambar yang Diunggah")
    st.image(img, use_column_width=True)

    img_np = np.array(img)
    img_np = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    img_np = img_np / 255.0
    img_np = np.expand_dims(img_np, axis=0)

    # Predict
    pred = predict_soil(img_np)[0]
    class_idx = np.argmax(pred)
    confidence = float(np.max(pred) * 100)

    soil_type = class_names[class_idx]
    data = tanaman_cocok.get(soil_type, None)

    # =======================
    #  HASIL PREDIKSI CARD
    # =======================
    st.markdown("""
    <br>
    <div style="background-color:#E8F5E9; padding:20px; border-radius:12px; border-left:8px solid #43A047;">
        <h2 style="color:#1B5E20;">🔍 Hasil Prediksi</h2>
    """, unsafe_allow_html=True)

    st.markdown(
        f"""
        <h3 style="color:#2E7D32;">{data['emoji']} Jenis Tanah: <b>{soil_type}</b></h3>
        <p style="font-size:16px;">Tingkat keyakinan model: <b>{confidence:.2f}%</b></p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div><br>", unsafe_allow_html=True)

    # =======================
    #  INFORMASI DAN REKOMENDASI
    # =======================
    st.markdown("""
    <div style="background-color:#F1F8E9; padding:20px; border-radius:12px; border-left:8px solid #8BC34A;">
        <h3 style="color:#33691E;">🌿 Rekomendasi Tanaman</h3>
    """, unsafe_allow_html=True)

    st.markdown(
        f"""
        <p style="font-size:16px;"><b>Deskripsi:</b> {data['deskripsi']}</p>
        <p style="font-size:16px;"><b>Tanaman yang cocok:</b> {data['tanaman']}</p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)
