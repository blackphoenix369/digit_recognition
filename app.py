import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# 1. Page Configuration
st.set_page_config(page_title="Handwritten Digit Recognizer", layout="centered")

# 2. Load the Model
@st.cache_resource
def load_digit_model():
    # Ensure "digit_model.h5" is in the same directory
    return tf.keras.models.load_model("digit_model.h5")

try:
    model = load_digit_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. UI Header
st.title("🧠 Handwritten Digit Recognition")
st.markdown("""
Draw a digit in the box below or upload an image. 
The model works best when the digit is centered and clear.
""")

# 4. Input Options (Tabs)
tab1, tab2 = st.tabs(["🖌️ Draw Digit", "📤 Upload Image"])

with tab1:
    st.write("Draw inside the box:")
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with tab2:
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# 5. Image Processing Function
def process_image(raw_image, invert=False):
    """
    Standardizes input to MNIST format: 28x28, Grayscale, 
    White digit on Black background.
    """
    # Convert to Grayscale
    img = raw_image.convert("L")
    
    # Invert colors if necessary (for uploads on white paper)
    if invert:
        img = ImageOps.invert(img)
    
    # Resize to model input size
    img = img.resize((28, 28))
    
    # Normalize pixel values (0-1)
    img_array = np.array(img) / 255.0
    
    # Reshape for MLP (1, 28, 28) 
    # NOTE: If your model is a Dense/Flat MLP, use .reshape(1, 784)
    img_tensor = img_array.reshape(1, 28, 28)
    
    return img_tensor, img

# 6. Logic to Determine Input Source
img_tensor = None

if uploaded_file:
    # Processing uploaded file (assumes black ink on white background)
    input_img = Image.open(uploaded_file)
    img_tensor, processed_preview = process_image(input_img, invert=True)
    
elif canvas_result.image_data is not None:
    # Check if user has actually drawn something (sum of alpha channel > 0)
    if np.any(canvas_result.image_data[:, :, 3] > 0):
        # Canvas is already white on black, so invert=False
        input_img = Image.fromarray(np.uint8(canvas_result.image_data[:, :, 0:3]))
        img_tensor, processed_preview = process_image(input_img, invert=False)

# 7. Prediction and Visualization
if img_tensor is not None:
    st.divider()
    cols = st.columns([1, 2])
    
    with cols[0]:
        st.write("### Input")
        st.image(processed_preview, caption="What the model sees", width=150)

    with cols[1]:
        # Perform Prediction
        prediction = model.predict(img_tensor)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        st.write(f"### Prediction: **{predicted_digit}**")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.progress(int(confidence))

        # Show probability chart
        chart_data = pd.DataFrame(
            prediction[0], 
            index=[str(i) for i in range(10)], 
            columns=["Probability"]
        )
        st.bar_chart(chart_data)
else:
    st.info("Please draw a digit or upload an image to see the prediction.")