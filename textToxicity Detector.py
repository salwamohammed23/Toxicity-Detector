import streamlit as st
from PIL import Image
import numpy as np
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
import torch

# Page configuration
st.set_page_config(page_title="🛡️ Smart Content Classifier", layout="centered")
st.title("🛡️ Content Safety & Toxicity Classifier")

# Load all models and resources once
@st.cache_resource
def load_all_models():
    # FLAN-T5
    flan_pipe = pipeline("text2text-generation", model="google/flan-t5-base")

    # BLIP
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blip_model.to(device)

    # Local Model
    keras_model = tf.keras.models.load_model("Model/text_classifier_cnn.h55")
    with open("Model/tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)

    return flan_pipe, processor, blip_model, device, keras_model, tokenizer

# Load models
try:
    flan_pipe, blip_processor, blip_model, torch_device, keras_model, tokenizer = load_all_models()
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()

# Define local model class labels
class_names = [
    "Safe", "Violent Crimes", "Elections", "Sex-Related Crimes", "Unsafe",
    "Non-Violent Crimes", "Child Sexual Exploitation", "Unknown S-Type", "Suicide & Self-Harm"
]
id2label = {str(i): label for i, label in enumerate(class_names)}

# FLAN-T5 classification function
def classify_with_flan(text):
    prompt = (
        f"Is the following content Safe or Unsafe?\n\n"
        f"\"{text}\"\n\n"
        f"Answer with only one word: Safe or Unsafe."
    )
    result = flan_pipe(prompt, max_new_tokens=10)
    return result[0]['generated_text'].strip()

# Image caption generation with BLIP
def generate_caption(image_file):
    image = Image.open(image_file).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt").to(torch_device)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# Classification using local model
def classify_text_local(text, model, tokenizer, id2label, max_length=100):
    sequence = tokenizer.texts_to_sequences([text])
    if not sequence[0]:
        return {
            'text': text,
            'predicted_label': "Unknown",
            'predicted_id': -1,
            'probabilities': {"Unknown": 1.0}
        }
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)
    predicted_id = int(np.argmax(prediction))
    predicted_label = id2label.get(str(predicted_id), f"class_{predicted_id}")
    probabilities = {
        id2label.get(str(i), f"class_{i}"): float(prob)
        for i, prob in enumerate(prediction[0])
    }
    return {
        'text': text,
        'predicted_label': predicted_label,
        'predicted_id': predicted_id,
        'probabilities': probabilities
    }

# ====== User Interface ======
input_type = st.selectbox("🔽 Choose input type:", ["Select...", "Text", "Image"])

user_text = ""
caption = ""

if input_type == "Text":
    user_text = st.text_area("✍️ Enter your text here", height=150)

elif input_type == "Image":
    uploaded_image = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

# Analyze button
if st.button("🔍 Analyze"):
    if input_type == "Text" and user_text.strip():
        with st.spinner("Initial classification with FLAN-T5..."):
            flan_result = classify_with_flan(user_text)

        st.subheader("🔎 FLAN-T5: Preliminary Classification")
        st.write(f"📌 **Text:** {user_text}")
        st.write(f"🔒 **Result:** `{flan_result}`")

        if flan_result.lower() == "unsafe":
            with st.spinner("Deep analysis using the local model..."):
                result = classify_text_local(user_text, keras_model, tokenizer, id2label)

            st.subheader("🚨 Detailed Classification (Local Model):")
            st.metric("📌 Classification:", result["predicted_label"])
            st.bar_chart(result["probabilities"])

    elif input_type == "Image" and uploaded_image:
        with st.spinner("🖼️ Generating caption for image..."):
            caption = generate_caption(uploaded_image)
        st.image(uploaded_image, caption="📷 Uploaded Image", use_container_width=True) 
        st.write(f"✏️ **Generated Caption:** {caption}")

        with st.spinner("Classifying caption with FLAN-T5..."):
            flan_result = classify_with_flan(caption)

        st.subheader("🔎 FLAN-T5: Preliminary Classification")
        st.write(f"🔒 **Result:** `{flan_result}`")

        if flan_result.lower() == "unsafe":
            with st.spinner("Deep analysis using the local model..."):
                result = classify_text_local(caption, keras_model, tokenizer, id2label)

            st.subheader("🚨 Detailed Classification (Local Model):")
            st.metric("📌 Classification:", result["predicted_label"])
            st.bar_chart(result["probabilities"])

    else:
        st.warning("⚠️ Please enter text or upload an image.")

# Footer
st.markdown("---")
st.caption("🛡️ Developed by Salwa — Streamlit, Transformers, TensorFlow")
