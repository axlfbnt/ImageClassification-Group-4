import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

from config import APP_TITLE, APP_SUBTITLE, MODEL_PATHS
from model.model_loader import get_model
from model.labels import get_class_names
from utils.image_utils import preprocess_image, predict_image

# Page config
st.set_page_config(page_title=APP_TITLE, layout="centered")

# Sidebar
st.sidebar.title("🔍 CIFAR-10 Classifier")
st.sidebar.markdown("Upload image to classify using trained CNN")

# Model selector
selected_model = st.sidebar.selectbox("🧠 Choose Model", list(MODEL_PATHS.keys()))

# Tambahan: Sidebar Model Info
with st.sidebar.expander("ℹ️ Model Info"):
    st.markdown(f"**Model Terpilih:** <span style='color:lightgreen'>{selected_model}</span>", unsafe_allow_html=True)
    st.write("Model ini mengenali 10 kelas dari CIFAR-10:")
    class_list = get_class_names()
    class_tags = ", ".join([f"`{cls}`" for cls in class_list])
    st.markdown(f"{class_tags}")

# Main title
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# Upload image
uploaded_file = st.file_uploader("Upload Image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    if st.button("🔎 Predict"):
        with st.spinner(f"Loading `{selected_model}` and classifying..."):
            model = get_model(selected_model)
            input_array = preprocess_image(image)
            label, confidence, probabilities = predict_image(model, input_array)

        THRESHOLD = 0.6  # Confidence threshold

        if confidence < THRESHOLD:
            st.warning("🤔 Gambar tidak dikenali atau tidak sesuai kelas manapun.")
        else:
            with st.expander("📌 Klik untuk melihat hasil prediksi"):
                st.subheader("🎯 Hasil Klasifikasi")
                st.success(f"Gambar ini diprediksi sebagai **{label}**")
                st.metric(label="Tingkat Keyakinan", value=f"{confidence:.2%}")

                # Bar chart visualisasi confidence
                df_conf = pd.DataFrame({
                    "Class": get_class_names(),
                    "Confidence": probabilities
                })
                fig = px.bar(df_conf.sort_values("Confidence", ascending=False),
                             x="Class", y="Confidence", color="Class",
                             title="Confidence untuk Setiap Kelas",
                             text=df_conf["Confidence"].apply(lambda x: f"{x:.2%}"))
                st.plotly_chart(fig, use_container_width=True)
