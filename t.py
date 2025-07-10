import streamlit as st
import torch
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

# 1. Application Setup
st.set_page_config(
    page_title="Harmful Content Detection Model",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("⚠️ Harmful Content Detector using LoRA Technology")

# 2. Enhanced Model Loading
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    try:
        model_path = "Model/lora_distilbert_toxic_final"
        
        # Required files with custom error messages
        required_files = {
            'adapter_config.json': "LoRA configuration file",
            'adapter_model.safetensors': "Fine-tuned model weights",
            'tokenizer_config.json': "Tokenizer settings",
            'vocab.txt': "Vocabulary dictionary"
        }
        
        # Verify all files exist
        missing_files = []
        for file, desc in required_files.items():
            if not os.path.exists(os.path.join(model_path, file)):
                missing_files.append(f"- {desc} ({file})")
        
        if missing_files:
            raise FileNotFoundError(
                "The following files are missing:\n" + "\n".join(missing_files)
            )
        
        # Load model components with progress indicators
        with st.spinner("Loading model configuration..."):
            config = PeftConfig.from_pretrained(model_path)
        
        with st.spinner("Loading base model..."):
            base_model = AutoModelForSequenceClassification.from_pretrained(
                config.base_model_name_or_path,
                num_labels=9,  # Changed from 8 to 9 to match trained model
                return_dict=True,
                ignore_mismatched_sizes=True,
                device_map="auto"
            )
        
        with st.spinner("Applying LoRA adapters..."):
            model = PeftModel.from_pretrained(base_model, model_path)
        
        with st.spinner("Loading tokenizer..."):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model, tokenizer
        
    except Exception as e:
        st.error("## Error loading model")
        st.error(str(e))
        st.error("""
**Possible actions:**
1. Verify model folder exists in correct path
2. Check all required files are present
3. Confirm correct library versions are installed
4. Review error log for details""")
        return None, None

model, tokenizer = load_model()

# 3. Enhanced User Interface
if model and tokenizer:
    st.sidebar.success("Model loaded successfully!")
    
    # Label definitions with visual indicators
    LABELS = {
        "Safe": {"emoji": "✅", "color": "green"},
        "Hate Speech": {"emoji": "💢", "color": "red"},
        "Insult": {"emoji": "🗯️", "color": "orange"},
        "Threat": {"emoji": "⚠️", "color": "red"},
        "Racist": {"emoji": "🚫", "color": "red"},
        "Sexual": {"emoji": "🔞", "color": "red"},
        "Incitement": {"emoji": "🔥", "color": "orange"},
        "Other": {"emoji": "❓", "color": "gray"},
        "Self-Harm": {"emoji": "💔", "color": "red"}  # Added 9th category
    }
    
    with st.form("classification_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            text = st.text_area(
                "**Enter text to analyze:**",
                height=200,
                placeholder="Paste text here...",
                help="Enter any text to analyze its content"
            )
        with col2:
            st.markdown("### Settings")
            max_length = st.slider("Max length", 128, 512, 256)
            threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.7, 0.05)
        
        submitted = st.form_submit_button("**Analyze Text**", use_container_width=True)
        
        if submitted and text:
            with st.spinner("Analyzing text..."):
                try:
                    # Tokenization with error handling
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=max_length
                    ).to(model.device)
                    
                    # Prediction
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        pred_idx = torch.argmax(probs).item()
                        confidence = probs[0][pred_idx].item()
                    
                    # Display results
                    st.subheader("📊 Analysis Results")
                    
                    pred_label = list(LABELS.keys())[pred_idx]
                    label_info = LABELS[pred_label]
                    
                    # Main result card
                    if confidence > threshold:
                        st.success(f"""
                        ## {label_info['emoji']} Classification: **{pred_label}**  
                        **Confidence Level**: {confidence:.2%}  
                        **Assessment**: {"Dangerous" if pred_idx > 0 else "Safe"}
                        """)
                    else:
                        st.warning(f"""
                        ## ⚠️ Inconclusive Classification  
                        **Most Likely Category**: {pred_label}  
                        **Confidence**: {confidence:.2%} (below {threshold:.0%} threshold)
                        """)
                    
                    # Probability distribution
                    st.markdown("### Probability Distribution by Category")
                    for i, (label, prob) in enumerate(zip(LABELS.keys(), probs[0])):
                        prob_value = prob.item()
                        label_info = LABELS[label]
                        
                        cols = st.columns([1, 3, 1])
                        cols[0].markdown(f"**{label_info['emoji']} {label}**")
                        cols[1].progress(
                            prob_value,
                            text=f"{prob_value:.2%}"
                        )
                        cols[2].markdown(f"`{prob_value:.2%}`")
                    
                    # Warning for low confidence results
                    if confidence < threshold:
                        st.warning("""
                        **Note**: Results below threshold may indicate:
                        - Ambiguous text
                        - Unclear language
                        - Indeterminate context
                        """)
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.error("Text may be too long or invalid")

# 4. Enhanced Sidebar Information
st.sidebar.markdown("## 🛠️ Technical Information")
st.sidebar.info("""
**Model Details:**
- **Base Model**: DistilBERT-base-uncased
- **Technique**: LoRA (Low-Rank Adaptation)
- **Number of Categories**: 9
- **Model Size**: ~70MB (with LoRA)

**Capabilities:**
- Harmful content detection
- Toxicity classification
- Hate speech analysis
""")

st.sidebar.markdown("## 📊 Statistics")
if model:
    st.sidebar.metric("Number of Categories", len(LABELS))
    st.sidebar.metric("Tokenizer Vocabulary", f"{len(tokenizer):,} tokens")

# 5. Page Footer
st.markdown("---")
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f1f1f1;
    color: #555;
    text-align: center;
    padding: 10px;
    font-size: 12px;
}
</style>
<div class="footer">
    <p>Developed using 🤗 Transformers and PEFT | Version v1.1.0</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
