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

# إعداد التطبيق
st.set_page_config(
    page_title="محلل سلامة المحتوى المتقدم",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🛡️ محلل سلامة المحتوى المتقدم")

# تحميل النماذج
@st.cache_resource(show_spinner="جاري تحميل النماذج...")
def load_models():
    try:
        # نموذج وصف الصور
        with st.spinner("جاري تحميل نموذج وصف الصور..."):
            blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # نموذج الفحص الأولي
        with st.spinner("جاري تحميل نموذج الفحص الأولي..."):
            flan_pipe = pipeline("text2text-generation", model="google/flan-t5-base")
        
        # نموذج التحليل التفصيلي
        with st.spinner("جاري تحميل نموذج التحليل التفصيلي..."):
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
        
        # نقل النماذج لـ GPU إذا متاح
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blip_model.to(device)
        lora_model.to(device)
        
        return blip_processor, blip_model, flan_pipe, lora_model, tokenizer, device
    
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل النماذج: {str(e)}")
        return None, None, None, None, None, None

# تعريف التصنيفات
LABELS = {
    0: {"name": "آمن", "emoji": "✅", "color": "green"},
    1: {"name": "خطاب كراهية", "emoji": "💢", "color": "red"},
    2: {"name": "إهانة", "emoji": "🗯️", "color": "orange"},
    3: {"name": "تهديد", "emoji": "⚠️", "color": "red"},
    4: {"name": "عنصري", "emoji": "🚫", "color": "red"},
    5: {"name": "جنسي", "emoji": "🔞", "color": "red"},
    6: {"name": "تحريض", "emoji": "🔥", "color": "orange"},
    7: {"name": "أخرى", "emoji": "❓", "color": "gray"},
    8: {"name": "إيذاء ذاتي", "emoji": "💔", "color": "red"}
}

# واجهة المستخدم
def main():
    blip_processor, blip_model, flan_pipe, lora_model, tokenizer, device = load_models()
    
    # زر إعادة تحميل الصفحة
    if st.button("🔄 تحديث الصفحة"):
        st.experimental_rerun()
    
    input_type = st.radio(
        "اختر نوع المحتوى:",
        ["نص", "صورة"],
        horizontal=True,
        key="input_type"
    )
    
    if input_type == "صورة":
        uploaded_file = st.file_uploader(
            "رفع صورة للتحليل:",
            type=["jpg", "jpeg", "png"],
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            # عرض الصورة مباشرة
            st.image(uploaded_file, caption="الصورة المرفوعة", use_column_width=True)
            
            if st.button("تحليل الصورة", key="analyze_image"):
                with st.spinner("جاري تحليل الصورة..."):
                    try:
                        # توليد الوصف
                        raw_image = Image.open(uploaded_file).convert("RGB")
                        inputs = blip_processor(raw_image, return_tensors="pt").to(device)
                        out = blip_model.generate(**inputs)
                        caption = blip_processor.decode(out[0], skip_special_tokens=True)
                        
                        st.success(f"**التسمية التوضيحية:** {caption}")
                        
                        # التحليل التفصيلي
                        inputs = tokenizer(
                            caption,
                            return_tensors="pt",
                            truncation=True,
                            padding=True,
                            max_length=256
                        ).to(device)
                        
                        with torch.no_grad():
                            outputs = lora_model(**inputs)
                            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        
                        pred_idx = torch.argmax(probs).item()
                        confidence = probs[0][pred_idx].item()
                        label = LABELS[pred_idx]
                        
                        st.subheader("📊 نتائج التحليل")
                        st.markdown(f"""
                        <div style='background-color:#f0f0f0; padding:15px; border-radius:10px; border-left:5px solid {label["color"]}'>
                            <h3 style='color:{label["color"]}'>{label["emoji"]} التصنيف: <strong>{label["name"]}</strong></h3>
                            <p>مستوى الثقة: {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # عرض توزيع الاحتمالات
                        st.write("### توزيع الاحتمالات:")
                        for i, prob in enumerate(probs[0]):
                            label_info = LABELS[i]
                            cols = st.columns([1, 3, 1])
                            cols[0].markdown(f"**{label_info['emoji']} {label_info['name']}**")
                            cols[1].progress(prob.item(), text=f"{prob.item():.2%}")
                            cols[2].write(f"{prob.item():.2%}")
                            
                    except Exception as e:
                        st.error(f"حدث خطأ أثناء التحليل: {str(e)}")
    
    elif input_type == "نص":
        text_content = st.text_area(
            "أدخل النص للتحليل:",
            height=200,
            placeholder="الصق النص هنا...",
            key="text_input"
        )
        
        if st.button("تحليل النص", key="analyze_text") and text_content.strip():
            with st.spinner("جاري تحليل النص..."):
                try:
                    # نفس كود التحليل السابق للنص
                    pass
                except Exception as e:
                    st.error(f"حدث خطأ أثناء التحليل: {str(e)}")

if __name__ == "__main__":
    main()
