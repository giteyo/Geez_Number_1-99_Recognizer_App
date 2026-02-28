import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Geez Number Classifier (1 upto 99)", layout="centered")

@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model('my_cnn_model.h5')
        with open('classes.pkl', 'rb') as f:
            class_names = pickle.load(f)
        return model, class_names
    except:
        return None, None

model, class_names = load_assets()

# --- MAIN UI ---
st.title("🖼️ Geez Number Classifier (1 upto 99)")
st.write("Drag and drop an image below to analyze.")

# st.file_uploader natively supports Drag and Drop
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    
    # Preprocessing
    processed_img = image.convert('L').resize((100, 100))
    img_array = np.array(processed_img) / 255.0
    img_array = img_array.reshape(1, 100, 100, 1)

    if st.button('🚀 Analyze'):
        predictions = model.predict(img_array)
        score = np.max(predictions)
        label = class_names[np.argmax(predictions)]

        st.success(f"### Result: {label}")
        st.write(f"Confidence: {score*100:.2f}%")

        # --- DOWNLOAD REPORT FEATURE ---
        report_text = f"""
        Classification Report
        ---------------------
        Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        File Name: {uploaded_file.name}
        Detected Class: {label}
        Confidence: {score*100:.2f}%
        """
        
        st.download_button(
            label="📥 Download Results as TXT",
            data=report_text,
            file_name=f"report_{label}.txt",
            mime="text/plain"
        )

        # Show the breakdown
        chart_data = dict(zip(class_names, predictions[0].astype(float)))
        st.bar_chart(chart_data)