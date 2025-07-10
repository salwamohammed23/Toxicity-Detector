import streamlit as st
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BlipProcessor,
    BlipForConditionalGeneration,
    pipeline
)
from peft import PeftModel, PeftConfig
from PIL import Image
import torch
import os

# 1. Application Setup
st.set_page_config(
    page_title="Advanced Content Safety Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🛡️ Advanced Content Safety Analyzer")

# 2. Load All Models
@st.cache_resource(show_spinner="Loading all models...")
def load_models():
    try:
        # Load BLIP image captioning model
        with st.spinner("Loading image captioning model..."):
            blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Load FLAN-T5 for initial safety check
        with st.spinner("Loading initial safety classifier..."):
            flan_pipe = pipeline("text2text-generation", model="google/flan-t5-base")
        
        # Load fine-tuned LoRA model for detailed analysis
        with st.spinner("Loading detailed content analyzer..."):
            model_path = "Model/lora_distilbert_toxic_final"
            config = PeftConfig.from_pretrained(model_path)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                config.base_model_name_or_path,
                num_labels=9,
                return_dict=True,
                ignore_mismatched_sizes=True
            )
            lora_model = PeftModel.from_pretrained(base_model, model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Move models to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blip_model.to(device)
        lora_model.to(device)
        
        return blip_processor, blip_model, flan_pipe, lora_model, tokenizer, device
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None, None

blip_processor, blip_model, flan_pipe, lora_model, tokenizer, device = load_models()

# 3. Helper Functions
def generate_caption(image):
    """Generate caption from image using BLIP model"""
    raw_image = Image.open(image).convert("RGB")
    inputs = blip_processor(raw_image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def initial_safety_check(text):
    """Quick safety check using FLAN-T5"""
    prompt = f"Is this Safe or Unsafe? \"{text}\" Answer with one word: Safe or Unsafe."
    result = flan_pipe(prompt, max_new_tokens=10)
    return result[0]['generated_text'].strip().lower()

def detailed_analysis(text):
    """Detailed content analysis using LoRA model"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)
    
    with torch.no_grad():
        outputs = lora_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    return probs[0].tolist()

# 4. Label Definitions
LABELS = {
    0: {"name": "Safe", "emoji": "✅", "color": "green"},
    1: {"name": "Hate Speech", "emoji": "💢", "color": "red"},
    2: {"name": "Insult", "emoji": "🗯️", "color": "orange"},
    3: {"name": "Threat", "emoji": "⚠️", "color": "red"},
    4: {"name": "Racist", "emoji": "🚫", "color": "red"},
    5: {"name": "Sexual", "emoji": "🔞", "color": "red"},
    6: {"name": "Incitement", "emoji": "🔥", "color": "orange"},
    7: {"name": "Other", "emoji": "❓", "color": "gray"},
    8: {"name": "Self-Harm", "emoji": "💔", "color": "red"}
}

# 5. User Interface
with st.form("content_form"):
    col1, col2 = st.columns([3, 1])
    
    with col1:
        input_type = st.radio(
            "Select content type:",
            ["Text", "Image"],
            horizontal=True
        )
        
        if input_type == "Text":
            content = st.text_area(
                "Enter text to analyze:",
                height=200,
                placeholder="Paste text here..."
            )
        else:
            content = st.file_uploader(
                "Upload image to analyze:",
                type=["jpg", "jpeg", "png"]
            )
    
    submitted = st.form_submit_button("Analyze Content", use_container_width=True)

# 6. Processing Pipeline
if submitted and (content or (input_type == "Text" and content.strip())):
    with st.spinner("Analyzing content..."):
        try:
            # Stage 1: Get text content (either direct or from image caption)
            if input_type == "Image":
                st.image(content, caption="Uploaded Image", use_column_width=True)
                caption = generate_caption(content)
                st.success(f"Generated caption: {caption}")
                text_to_analyze = caption
            else:
                text_to_analyze = content
            
            # Stage 2: Initial safety check (quick filter)
            initial_check = initial_safety_check(text_to_analyze)
            
            if "unsafe" in initial_check:
                st.error("## ❌ Initial Safety Check: Unsafe Content")
                st.error("This content was flagged as potentially unsafe in initial screening.")
                st.stop()
            
            # Stage 3: Detailed analysis
            st.success("## ✅ Initial Safety Check: Passed")
            st.info("Proceeding with detailed analysis...")
            
            probs = detailed_analysis(text_to_analyze)
            pred_idx = max(range(len(probs)), key=lambda i: probs[i])
            confidence = probs[pred_idx]
            label = LABELS[pred_idx]
            
            # Display results
            st.subheader("📊 Detailed Analysis Results")
            st.success(f"""
            ## {label['emoji']} Classification: **{label['name']}**  
            **Confidence Level**: {confidence:.2%}  
            **Assessment**: {"Potentially harmful" if pred_idx > 0 else "Safe"}
            """)
            
            # Probability distribution
            st.markdown("### Probability Distribution by Category")
            for i, prob in enumerate(probs):
                label_info = LABELS[i]
                cols = st.columns([1, 3, 1])
                cols[0].markdown(f"**{label_info['emoji']} {label_info['name']}**")
                cols[1].progress(
                    prob,
                    text=f"{prob:.2%}"
                )
                cols[2].markdown(f"`{prob:.2%}`")
        
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

elif submitted:
    st.warning("Please provide content to analyze")

# 7. Sidebar Information
st.sidebar.markdown("## 🛠️ Technical Information")
st.sidebar.info("""
**Model Pipeline:**
1. BLIP (Image Captioning)
2. FLAN-T5 (Initial Safety Check)
3. LoRA DistilBERT (Detailed Analysis)

**Safety Categories:**
- ✅ Safe
- 💢 Hate Speech
- 🗯️ Insult
- ⚠️ Threat
- 🚫 Racist
- 🔞 Sexual
- 🔥 Incitement
- ❓ Other
- 💔 Self-Harm
""")

# 8. Footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    text-align: center;
    padding: 10px;
    font-size: 12px;
}
</style>
<div class="footer">
    <p>Advanced Content Safety Analyzer | Multi-Modality AI Protection</p>
</div>
""", unsafe_allow_html=True)
