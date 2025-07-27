import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np
import pandas as pd
import plotly.express as px

from config import APP_TITLE, APP_SUBTITLE, MODEL_PATHS
from model.model_loader import get_model
from model.labels import get_class_names
from utils.image_utils import preprocess_image, predict_image

# Konfigurasi halaman
st.set_page_config(page_title=APP_TITLE, layout="centered")

# Sidebar - Judul dan penjelasan
st.sidebar.title("🔍 CIFAR-10 Classifier")
st.sidebar.markdown("Upload gambar untuk diklasifikasi menggunakan model CNN terlatih.")

# Sidebar - Pilih model
selected_model = st.sidebar.selectbox("🧠 Pilih Model", list(MODEL_PATHS.keys()))

# Sidebar - Informasi model langsung
model_descriptions = {
    "Simple CNN": "Model CNN dengan 8 layer konvolusi bertingkat dari 32 hingga 256 filter, "
                  "menggunakan batch normalization dan dropout bertahap. "
                  "Dirancang untuk akurasi lebih baik pada CIFAR-10, meskipun tetap lebih ringan dari model pretrained.",
    "VGG16 (Transfer Learn)": "Model VGG16 pretrained dengan fine-tuning untuk klasifikasi CIFAR-10. "
                              "Akurasi dapat lebih tinggi, cocok untuk inference dengan resource memadai."
}

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model Terpilih:** <span style='color:lightgreen'>{selected_model}</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"📝 {model_descriptions[selected_model]}")
st.sidebar.markdown("📚 Model ini mengenali 10 kelas CIFAR-10:")
class_tags = ", ".join([f"`{cls}`" for cls in get_class_names()])
st.sidebar.markdown(class_tags)
st.sidebar.markdown("---")

# Main title
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# Upload gambar
uploaded_file = st.file_uploader("Upload Gambar (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Gambar yang Diupload", use_container_width=True)

        if st.button("🔎 Prediksi"):
            with st.spinner(f"Loading model `{selected_model}` dan mengklasifikasi..."):
                model = get_model(selected_model)
                input_array = preprocess_image(image)
                label, confidence, probabilities = predict_image(model, input_array)

            THRESHOLD = 0.6  # Confidence threshold

            if confidence < THRESHOLD:
                st.warning("🤔 Gambar tidak dikenali atau tidak sesuai kelas manapun.")
            else:
                st.subheader("🎯 Hasil Klasifikasi")
                st.success(f"Gambar ini diprediksi sebagai **{label}**")
                st.metric(label="Tingkat Keyakinan", value=f"{confidence:.2%}")

                # Visualisasi confidence
                df_conf = pd.DataFrame({
                    "Class": get_class_names(),
                    "Confidence": probabilities
                })

                df_sorted = df_conf.sort_values("Confidence", ascending=False).assign(
                    ConfidenceText=lambda df: df["Confidence"].apply(lambda x: f"{x:.2%}")
                )

                fig = px.bar(df_sorted,
                             x="Class", y="Confidence", color="Class",
                             title="Confidence untuk Setiap Kelas",
                             text="ConfidenceText")
                fig.update_traces(textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig, use_container_width=True)

    except UnidentifiedImageError:
        st.error("❌ File yang diunggah bukan gambar yang valid. Harap unggah file JPG atau PNG.")
    except Exception as e:
        st.error(f"❌ Terjadi error saat memproses gambar: {str(e)}")