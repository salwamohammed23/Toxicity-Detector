import streamlit as st
import torch
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

# 1. تهيئة التطبيق
st.set_page_config(
    page_title="نموذج كشف المحتوى الضار",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("⚠️ كاشف المحتوى الضار باستخدام تقنية LoRA")

# 2. تحميل النموذج - نسخة محسنة
@st.cache_resource(show_spinner="جاري تحميل النموذج...")
def load_model():
    try:
        model_path = "Model/lora_distilbert_toxic_final"
        
        # قائمة بالملفات المطلوبة مع رسائل خطأ مخصصة
        required_files = {
            'adapter_config.json': "ملف تكوين LoRA الأساسي",
            'adapter_model.safetensors': "أوزان النموذج المحسنة",
            'tokenizer_config.json': "إعدادات Tokenizer",
            'vocab.txt': "قاموس المفردات"
        }
        
        # التحقق من وجود جميع الملفات
        missing_files = []
        for file, desc in required_files.items():
            if not os.path.exists(os.path.join(model_path, file)):
                missing_files.append(f"- {desc} ({file})")
        
        if missing_files:
            raise FileNotFoundError(
                "الملفات التالية مفقودة:\n" + "\n".join(missing_files)
            )
        # تحميل مكونات النموذج مع شريط تقدم
        with st.spinner("جاري تحميل إعدادات النموذج..."):
            config = PeftConfig.from_pretrained(model_path)
        
        with st.spinner("جاري تحميل النموذج الأساسي..."):
            base_model = AutoModelForSequenceClassification.from_pretrained(
                config.base_model_name_or_path,
                num_labels=8,
                return_dict=True,
                ignore_mismatched_sizes=True,
                device_map="auto"
            )
        
        with st.spinner("جاري تطبيق ضبط LoRA..."):
            model = PeftModel.from_pretrained(base_model, model_path)
        
        with st.spinner("جاري تحميل Tokenizer..."):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model, tokenizer
        
    except Exception as e:
        st.error("## حدث خطأ في تحميل النموذج")
        st.error(str(e))
        st.error("""
**الإجراءات الممكنة:**
1. تأكد من وجود مجلد النموذج في المسار الصحيح
2. تحقق من وجود جميع الملفات المطلوبة
3. تأكد من تثبيت الإصدارات الصحيحة للمكتبات
4. راجع سجل الأخطاء لمزيد من التفاصيل""")
        return None, None

model, tokenizer = load_model()

# 3. واجهة المستخدم المحسنة
if model and tokenizer:
    st.sidebar.success("تم تحميل النموذج بنجاح!")
    
    # تعريف الفئات مع ألوان توضيحية
    LABELS = {
        "غير سام": {"emoji": "✅", "color": "green"},
        "كراهية": {"emoji": "💢", "color": "red"},
        "إهانة": {"emoji": "🗯️", "color": "orange"},
        "تهديد": {"emoji": "⚠️", "color": "red"},
        "عنصري": {"emoji": "🚫", "color": "red"},
        "جنسي": {"emoji": "🔞", "color": "red"},
        "تحريض": {"emoji": "🔥", "color": "orange"},
        "أخرى": {"emoji": "❓", "color": "gray"}
    }
    
    with st.form("classification_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            text = st.text_area(
                "**أدخل النص المراد تحليله:**",
                height=200,
                placeholder="الصق النص هنا...",
                help="يمكنك إدخال أي نص لتحليل محتواه"
            )
        with col2:
            st.markdown("### إعدادات")
            max_length = st.slider("الحد الأقصى للطول", 128, 512, 256)
            threshold = st.slider("حد الثقة", 0.0, 1.0, 0.7, 0.05)
        
        submitted = st.form_submit_button("**بدء التحليل**", use_container_width=True)
        
        if submitted and text:
            with st.spinner("جاري تحليل النص..."):
                try:
                    # Tokenization مع معالجة الأخطاء
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=max_length
                    ).to(model.device)
                    
                    # التنبؤ
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        pred_idx = torch.argmax(probs).item()
                        confidence = probs[0][pred_idx].item()
                    
                    # عرض النتائج
                    st.subheader("📊 نتائج التحليل")
                    
                    pred_label = list(LABELS.keys())[pred_idx]
                    label_info = LABELS[pred_label]
                    
                    # بطاقة النتيجة الرئيسية
                    if confidence > threshold:
                        st.success(f"""
                        ## {label_info['emoji']} التصنيف: **{pred_label}**  
                        **مستوى الثقة**: {confidence:.2%}  
                        **التقييم**: { "خطير" if pred_idx > 0 else "آمن"}
                        """)
                    else:
                        st.warning(f"""
                        ## ⚠️ تصنيف غير حاسم  
                        **التصنيف الأكثر احتمالاً**: {pred_label}  
                        **الثقة**: {confidence:.2%} (أقل من الحد المطلوب {threshold:.0%})
                        """)
                    
                    # مخطط الاحتمالات
                    st.markdown("### توزيع الاحتمالات حسب الفئات")
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
                    
                    # تحذير إذا كانت أعلى نتيجة أقل من العتبة
                    if confidence < threshold:
                        st.warning("""
                        **ملاحظة**: النتيجة الأقل من الحد المطلوب قد تشير إلى:
                        - نص غامض
                        - لغة غير واضحة
                        - سياق غير محدد
                        """)
                        
                except Exception as e:
                    st.error(f"حدث خطأ أثناء التحليل: {str(e)}")
                    st.error("قد يكون النص طويلاً جداً أو غير صالح")

# 4. المعلومات الجانبية المحسنة
st.sidebar.markdown("## 🛠️ معلومات تقنية")
st.sidebar.info("""
**تفاصيل النموذج:**
- **النموذج الأساسي**: DistilBERT-base-uncased
- **التقنية**: LoRA (Low-Rank Adaptation)
- **عدد الفئات**: 8
- **حجم النموذج**: ~70MB (مع LoRA)

**إمكانيات النموذج:**
- كشف المحتوى الضار
- تصنيف أنواع السمية
- تحليل لغة الكراهية
""")

st.sidebar.markdown("## 📊 إحصائيات")
if model:
    st.sidebar.metric("عدد الفئات", len(LABELS))
    st.sidebar.metric("حجم Tokenizer", f"{len(tokenizer):,} مفردة")

# 5. تذييل الصفحة
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
    <p>تم تطويره باستخدام 🤗 Transformers و PEFT | إصدار v1.1.0</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
