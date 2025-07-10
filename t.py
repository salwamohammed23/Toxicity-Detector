import streamlit as st
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from peft import PeftModel, PeftConfig

# 1. تهيئة التطبيق
st.set_page_config(page_title="نموذج التصنيف النصي", layout="wide")
st.title("🎯 نموذج التصنيف باستخدام LoRA")

# 2. تحميل النموذج
@st.cache_resource
def load_model():
    try:
        # تحميل إعدادات LoRA
        config = PeftConfig.from_pretrained("Model/lora_distilbert_toxic_final")
        
        # تحميل النموذج الأساسي
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=8,
            return_dict=True
        )
        
        # تحميل نموذج LoRA
        model = PeftModel.from_pretrained(base_model, "Model/lora_distilbert_toxic_final")
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {str(e)}")
        return None, None

model, tokenizer = load_model()

# 3. واجهة المستخدم
if model and tokenizer:
    with st.form("text_classification"):
        text = st.text_area("أدخل النص للتصنيف:", height=150)
        
        if st.form_submit_button("صنّف") and text:
            try:
                # معالجة النص
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                # التنبؤ
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # عرض النتائج
                st.subheader("النتائج:")
                
                labels = ["فئة1", "فئة2", "فئة3", "فئة4", 
                         "فئة5", "فئة6", "فئة7", "فئة8"]
                
                for i, prob in enumerate(probs[0]):
                    st.progress(float(prob), text=f"{labels[i]}: {prob:.2%}")
                    
            except Exception as e:
                st.error(f"خطأ في التصنيف: {str(e)}")

# 4. معلومات جانبية
st.sidebar.markdown("## معلومات النموذج")
st.sidebar.info("""
- النموذج: DistilBERT + LoRA
- عدد الفئات: 8
- إصدار المكتبات:
  - Transformers: 4.33+
  - PEFT: 0.5+
""")
