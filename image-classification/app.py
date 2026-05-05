import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("model.h5")

labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# Page config
st.set_page_config(page_title="VisionAI Pro", page_icon="🧠", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.big-title {
    font-size: 55px;
    font-weight: bold;
    background: -webkit-linear-gradient(#00ffcc, #4CAF50);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.card {
    background: #111;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
.glow {
    background-color: #00ffcc;
    color: black;
    padding: 10px 20px;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<p class="big-title">🧠 VisionAI Pro</p>', unsafe_allow_html=True)
st.write("### 🚀 Smart Image Intelligence powered by Deep Learning")

# ---------------- HERO SECTION ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card">📊<br><b>Model Accuracy</b><br>~65%</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">⚡<br><b>Real-time AI</b><br>Instant Predictions</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">🧠<br><b>Deep Learning</b><br>CNN Model</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- DEMO BUTTON ----------------
st.subheader("🧪 Try AI Instantly")

if st.button("✨ Run Demo Prediction"):
    demo_img = np.random.rand(32,32,3)
    demo_img = np.expand_dims(demo_img, axis=0)

    pred = model.predict(demo_img)
    idx = np.argmax(pred)

    st.success(f"Demo Prediction: {labels[idx]}")

st.markdown("---")

# ---------------- UPLOAD ----------------
uploaded_files = st.file_uploader("📤 Upload Image(s)", type=["jpg","png","jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        col1, col2 = st.columns([1,2])

        with col1:
            image = Image.open(uploaded_file).resize((32,32))
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            index = np.argmax(prediction)
            confidence = prediction[0][index] * 100

            st.markdown(f"## 🎯 Prediction: **{labels[index].upper()}**")
            st.progress(int(confidence))
            st.write(f"Confidence: **{confidence:.2f}%**")

            # Top predictions
            st.subheader("🔍 Top 5 Predictions")
            top5 = np.argsort(prediction[0])[-5:][::-1]

            for i in top5:
                prob = prediction[0][i] * 100
                st.write(f"{labels[i]} - {prob:.2f}%")
                st.progress(int(prob))

        st.markdown("---")

# Footer
st.markdown("---")
st.caption("✨ VisionAI Pro | Built by SamWorks004 🚀")