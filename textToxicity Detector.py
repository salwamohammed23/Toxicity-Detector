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

# 1. إعداد التطبيق
st.set_page_config(
    page_title="محلل سلامة المحتوى المتقدم",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🛡️ محلل سلامة المحتوى المتقدم")

# 2. تحميل جميع النماذج
@st.cache_resource(show_spinner="جاري تحميل النماذج...")
def load_models():
    try:
        # تحميل نموذج إنشاء التسميات التوضيحية للصور
        with st.spinner("جاري تحميل نموذج وصف الصور..."):
            blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # تحميل نموذج الفحص الأولي للسلامة
        with st.spinner("جاري تحميل نموذج الفحص الأولي..."):
            flan_pipe = pipeline("text2text-generation", model="google/flan-t5-base")
        
        # تحميل نموذج التحليل التفصيلي
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
        
        # نقل النماذج إلى GPU إذا كان متاحًا
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blip_model.to(device)
        lora_model.to(device)
        
        return blip_processor, blip_model, flan_pipe, lora_model, tokenizer, device
    
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل النماذج: {str(e)}")
        return None, None, None, None, None, None

blip_processor, blip_model, flan_pipe, lora_model, tokenizer, device = load_models()

# 3. الدوال المساعدة
def generate_caption(image):
    """إنشاء تسمية توضيحية للصورة"""
    try:
        raw_image = Image.open(image).convert("RGB")
        inputs = blip_processor(raw_image, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs)
        return blip_processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"فشل في معالجة الصورة: {str(e)}")
        return None

def initial_safety_check(text):
    """فحص السلامة الأولي"""
    prompt = f"هل هذا المحتوى آمن أم غير آمن؟ \"{text}\" أجب بكلمة واحدة فقط: آمن أو غير آمن."
    result = flan_pipe(prompt, max_new_tokens=10)
    return result[0]['generated_text'].strip().lower()

def detailed_analysis(text):
    """تحليل تفصيلي للمحتوى"""
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

# 4. تعريف التصنيفات
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

# 5. واجهة المستخدم
with st.form("content_form"):
    input_type = st.radio(
        "اختر نوع المحتوى:",
        ["نص", "صورة"],
        horizontal=True
    )
    
    if input_type == "نص":
        content = st.text_area(
            "أدخل النص للتحليل:",
            height=200,
            placeholder="الصق النص هنا..."
        )
    else:
        content = st.file_uploader(
            "رفع صورة للتحليل:",
            type=["jpg", "jpeg", "png"]
        )
    
    submitted = st.form_submit_button("تحليل المحتوى", use_container_width=True)

# 6. معالجة المحتوى
if submitted:
    if (input_type == "صورة" and content is not None) or (input_type == "نص" and content.strip() != ""):
        with st.spinner("جاري تحليل المحتوى..."):
            try:
                # المرحلة 1: الحصول على النص (إما مباشرة أو من وصف الصورة)
                if input_type == "صورة":
                    st.image(content, caption="الصورة المرفوعة", use_column_width=True)
                    caption = generate_caption(content)
                    if caption is None:
                        st.stop()
                    
                    st.success(f"التسمية التوضيحية المولدة: {caption}")
                    text_to_analyze = caption
                else:
                    text_to_analyze = content
                
                # المرحلة 2: الفحص الأولي للسلامة
                initial_check = initial_safety_check(text_to_analyze)
                
                if "غير آمن" in initial_check:
                    st.error("## ❌ الفحص الأولي: محتوى غير آمن")
                    st.error("تم اكتشاف محتوى غير آمن في الفحص الأولي.")
                    st.stop()
                
                # المرحلة 3: التحليل التفصيلي
                st.success("## ✅ الفحص الأولي: محتوى آمن")
                st.info("جاري التحليل التفصيلي...")
                
                probs = detailed_analysis(text_to_analyze)
                pred_idx = max(range(len(probs)), key=lambda i: probs[i])
                confidence = probs[pred_idx]
                label = LABELS[pred_idx]
                
                # عرض النتائج
                st.subheader("📊 نتائج التحليل التفصيلي")
                st.success(f"""
                ## {label['emoji']} التصنيف: **{label['name']}**  
                **مستوى الثقة**: {confidence:.2%}  
                **التقييم**: {"غير آمن" if pred_idx > 0 else "آمن"}
                """)
                
                # توزيع الاحتمالات
                st.markdown("### توزيع الاحتمالات حسب الفئة")
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
                st.error(f"حدث خطأ أثناء التحليل: {str(e)}")
    else:
        st.warning("الرجاء إدخال نص أو رفع صورة")

# 7. معلومات إضافية
st.sidebar.markdown("## 🛠️ معلومات تقنية")
st.sidebar.info("""
**خطوات التحليل:**
1. BLIP (وصف الصور)
2. FLAN-T5 (فحص أولي)
3. DistilBERT (تحليل تفصيلي)

**فئات التصنيف:**
- ✅ آمن
- 💢 خطاب كراهية
- 🗯️ إهانة
- ⚠️ تهديد
- 🚫 عنصري
- 🔞 جنسي
- 🔥 تحريض
- ❓ أخرى
- 💔 إيذاء ذاتي
""")

# 8. تذييل الصفحة
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
    <p>محلل سلامة المحتوى المتقدم | حماية بواسطة الذكاء الاصطناعي</p>
</div>
""", unsafe_allow_html=True)
