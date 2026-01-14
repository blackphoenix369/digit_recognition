import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="CNN Digit Recognizer", layout="centered")

@st.cache_resource
def load_digit_model():
    return tf.keras.models.load_model("digit_model.h5")

try:
    model = load_digit_model()
except Exception as e:
    st.error(f"Model not found. Run your model.py script first! Error: {e}")
    st.stop()

st.title("🧠 CNN Digit Recognition")
tab1, tab2 = st.tabs(["🖌️ Draw", "📤 Upload"])

def process_image(raw_image, invert=False):
    img = raw_image.convert("L")
    if invert:
        img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    
    # --- CRITICAL CHANGE FOR CNN ---
    # We add the '1' at the end for the grayscale channel
    img_tensor = img_array.reshape(1, 28, 28, 1) 
    return img_tensor, img

with tab1:
    canvas_result = st_canvas(
        fill_color="#000000", stroke_width=20, stroke_color="#FFFFFF",
        background_color="#000000", width=280, height=280, 
        drawing_mode="freedraw", key="canvas"
    )

with tab2:
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

img_tensor = None
if uploaded_file:
    img_tensor, processed_preview = process_image(Image.open(uploaded_file), invert=True)
elif canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, 3] > 0):
    img_tensor, processed_preview = process_image(Image.fromarray(np.uint8(canvas_result.image_data[:, :, 0:3])), invert=False)

if img_tensor is not None:
    prediction = model.predict(img_tensor)
    digit = np.argmax(prediction)
    conf = np.max(prediction) * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(processed_preview, caption="Input", width=150)
    with col2:
        st.metric("Predicted Digit", digit)
        st.metric("Confidence", f"{conf:.1f}%")
    
    st.bar_chart(pd.DataFrame(prediction[0], index=list(range(10)), columns=["Prob"]))