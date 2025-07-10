import streamlit as st
import torch
import numpy as np
import os
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
        model_path = "Model/lora_distilbert_toxic_final"
        
        # التحقق من وجود الملفات المطلوبة
        required_files = ['adapter_config.json', 'adapter_model.safetensors']
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                raise FileNotFoundError(f"ملف {file} غير موجود في المسار المحدد")
        
        # تحميل إعدادات LoRA
        config = PeftConfig.from_pretrained(model_path)
        
        # تحميل النموذج الأساسي
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=8,
            return_dict=True,
            ignore_mismatched_sizes=True
        )
        
        # تحميل نموذج LoRA
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        
        return model, tokenizer
        
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {str(e)}")
        st.error("تأكد من:")
        st.error("1. وجود مجلد 'Model/lora_distilbert_toxic_final'")
        st.error("2. احتواء المجلد على ملفات adapter_config.json و adapter_model.bin")
        st.error("3. تثبيت جميع المكتبات المطلوبة")
        return None, None

model, tokenizer = load_model()

# 3. واجهة المستخدم
if model and tokenizer:
    with st.form("text_classification"):
        text = st.text_area("أدخل النص للتصنيف:", height=150, placeholder="اكتب النص هنا...")
        submitted = st.form_submit_button("صنّف")
        
        if submitted and text:
            try:
                # معالجة النص
                inputs = tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True,
                    max_length=512
                )
                
                # التنبؤ
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    preds = torch.argmax(probs, dim=1).item()
                
                # عرض النتائج
                st.subheader("🎯 نتائج التصنيف")
                
                # أسماء الفئات (يجب تعديلها حسب نموذجك)
                labels = [
                    "غير سام", "كراهية", "إهانة",
                    "تهديد", "عنصري", "جنسي",
                    "تحريض", "أخرى"
                ]
                
                # عرض التصنيف الرئيسي
                st.metric("التصنيف الرئيسي", labels[preds])
                
                # عرض احتمالات جميع الفئات
                st.write("**توزيع الاحتمالات:**")
                for i, (label, prob) in enumerate(zip(labels, probs[0])):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(f"{label}:")
                    with col2:
                        st.progress(float(prob), text=f"{prob:.2%}")
                        
            except Exception as e:
                st.error(f"حدث خطأ أثناء التصنيف: {str(e)}")
else:
    st.warning("تعذر تحميل النموذج. راجع رسائل الخطأ أعلاه.")

# 4. معلومات جانبية
st.sidebar.markdown("## معلومات النموذج")
st.sidebar.info("""
- **النموذج الأساسي:** DistilBERT
- **تقنية الضبط:** LoRA (PEFT)
- **عدد الفئات:** 8
- **المكتبات المستخدمة:**
  - Transformers: 4.33+
  - PEFT: 0.5+
  - PyTorch: 2.0+
""")

# 5. تذييل الصفحة
st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: 12px;
    text-align: center;
}
</style>
<div class="footer">
    تم التطوير باستخدام Streamlit و Hugging Face Transformers
</div>
""", unsafe_allow_html=True)
