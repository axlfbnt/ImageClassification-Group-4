import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from config import CLASS_NAMES

def preprocess_image(image: Image.Image):
    image = image.resize((32, 32)).convert("RGB")
    image_array = img_to_array(image).astype("float32")
    mean = 120.707
    std = 64.15
    normalized = (image_array - mean) / (std + 1e-7)
    return np.expand_dims(normalized, axis=0)

def predict_image(model, image_array):
    prediction = model.predict(image_array)[0]
    top_idx = np.argmax(prediction)
    top_label = CLASS_NAMES[top_idx]
    top_conf = prediction[top_idx]
    return top_label, top_conf, prediction