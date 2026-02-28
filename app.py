import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import arabic_reshaper
from bidi.algorithm import get_display

# --- وظائف معالجة اللغة العربية ---
def fix_text(text):
    if not text: return ""
    reshaped_text = arabic_reshaper.reshape(str(text))
    return get_display(reshaped_text)

# --- ثوابت النظام الجغرافي ---
ALL_PROVINCES = ['دمشق', 'ريف دمشق', 'حلب', 'حمص', 'حماة', 'اللاذقية', 'طرطوس', 'إدلب', 'دير الزور', 'الرقة', 'الحسكة', 'درعا', 'السويداء', 'القنيطرة']
ALL_SECTORS = ['خدمات تقنية', 'تجارة إلكترونية', 'زراعة حديثة', 'تصنيع غذائي', 'طاقة متجددة', 'صناعة هندسية']

PROVINCE_INFO = {
    'دمشق': {'cost': 1.4, 'feat': 'المركز المالي والإداري'},
    'ريف دمشق': {'cost': 1.2, 'feat': 'التجمع الصناعي والإنتاجي'},
    'حلب': {'cost': 1.15, 'feat': 'العاصمة الاقتصادية والصناعية'},
    'حمص': {'cost': 1.0, 'feat': 'عقدة اللوجستيات والطاقة'},
    'حماة': {'cost': 0.9, 'feat': 'المركز الزراعي والغذائي'},
    'اللاذقية': {'cost': 1.1, 'feat': 'المنفذ البحري والسياحي'},
    'طرطوس': {'cost': 1.0, 'feat': 'التجارة البحرية والنقل'},
    'درعا': {'cost': 0.9, 'feat': 'الإنتاج الزراعي المبكر'},
    'الحسكة': {'cost': 0.8, 'feat': 'سلة الغذاء والموارد الاستراتيجية'},
    'دير الزور': {'cost': 0.8, 'feat': 'إمكانات الطاقة والزراعة النهرية'},
    'الرقة': {'cost': 0.75, 'feat': 'مشاريع الري والإنتاج المكثف'},
    'إدلب': {'cost': 0.85, 'feat': 'التجارة والصناعات التحويلية'},
    'السويداء': {'cost': 0.95, 'feat': 'الزراعة الجبلية والسياحة البيئية'},
    'القنيطرة': {'cost': 0.7, 'feat': 'الإنتاج الحيواني والزراعي الواعد'}
}

# =================================================
# 1. محرك الذكاء الاصطناعي (AI Engine)
# =================================================
@st.cache_resource
def train_and_validate_model():
    try:
        df = pd.read_csv('startup_data.csv', encoding='utf-8-sig')
        le_p = LabelEncoder().fit(ALL_PROVINCES)
        le_s = LabelEncoder().fit(ALL_SECTORS)
        
        # تثبيت الترتيب لضمان استقرار النتائج
        target_labels = ['مخاطرة عالية', 'استثمار متوسط', 'فرصة واعدة']
        le_t = LabelEncoder()
        le_t.fit(target_labels)
        
        X = pd.DataFrame()
        mask = df['المحافظة'].isin(ALL_PROVINCES) & df['القطاع'].isin(ALL_SECTORS)
        df_clean = df[mask].copy()
        
        X['prov'] = le_p.transform(df_clean['المحافظة'])
        X['sect'] = le_s.transform(df_clean['القطاع'])
        X['cap'] = df_clean['رأس_المال_المطلوب_بالألف']
        X['risk'] = df_clean['مؤشر_المخاطرة']
        y = le_t.transform(df_clean['القرار_الاستثماري'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(max_depth=7, random_state=42) # زيادة العمق قليلاً للمرونة
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        conf_mat = confusion_matrix(y_test, model.predict(X_test))
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        return model, le_p, le_s, le_t, acc, conf_mat, cv_scores
    except Exception as e:
        return None, None, None, None, 0, None, []

clf, le_p, le_s, le_t, accuracy, conf_matrix, cv_results = train_and_validate_model()

# =================================================
# 2. واجهة المستخدم SyraVest
# =================================================
st.set_page_config(page_title="SyraVest AI Platform", layout="wide", page_icon="📈")

with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #1E88E5;'>SyraVest</h1>", unsafe_allow_html=True)
    st.image("syravest logo.png", width=200)
    st.markdown("---")
    
    st.subheader("👤 ملف المستثمر")
    risk_profile = st.radio("مدى تقبلك للمخاطرة:", ["متحفظ", "متوازن", "مغامر"])
    risk_map = {"متحفظ": 0.2, "متوازن": 0.45, "مغامر": 0.7}
    u_risk = risk_map[risk_profile]
    
    st.markdown("---")
    # تم تعديل سعر الصرف الافتراضي لعام 2026
    live_rate = st.number_input("سعر صرف الدولار (SYP)", value=11500.0)
    
    st.markdown("---")
    st.write("🌐 **تابعنا على:**")
    col_fb, col_in = st.columns(2)
    with col_fb:
        st.markdown("[![Facebook](https://img.icons8.com/color/48/facebook-new.png)](https://facebook.com/syravest)")
    with col_in:
        st.markdown("[![Instagram](https://img.icons8.com/fluency/48/instagram-new.png)](https://instagram.com/syravest)")
    
    st.markdown("---")
    with st.expander("⚠️ إخلاء مسؤولية"):
        st.caption("""
        النتائج استرشادية مبنية على خوارزميات تعلم الآلة. 
        لا تعتبر نصيحة مالية قطعية. 
        SyraVest تخلي مسؤوليتها عن أي قرارات استثمارية فردية.
        """)
    st.caption("SyraVest v26.5 © 2026")

# --- المحتوى الرئيسي ---
st.title("🛡️ منصة SyraVest لدعم القرار الاستثماري")
tabs = st.tabs(["🔍 تحليل الفرصة", "📍 مستشار المواقع", "💡 مبتكر الفرص", "📊 التقييم العلمي"])

with tabs[0]:
    c1, c2 = st.columns([1, 2])
    with c1:
        u_prov = st.selectbox("المحافظة", ALL_PROVINCES)
        u_sect = st.selectbox("القطاع", ALL_SECTORS)
        u_budget = st.number_input("الميزانية ($)", value=50000, step=5000)
        btn = st.button("تحليل ذكي")
    with c2:
        if btn and clf:
            # معايرة التضخم لنتائج أكثر واقعية
            inf = (live_rate / 11000) * PROVINCE_INFO[u_prov]['cost']
            adj_b = (u_budget / 1000) / inf
            pred = clf.predict([[le_p.transform([u_prov])[0], le_s.transform([u_sect])[0], adj_b, u_risk]])
            res = le_t.inverse_transform(pred)[0]
            
            st.subheader(f"النتيجة المتوقعة: {res}")
            st.info(f"📍 **الميزة:** {PROVINCE_INFO[u_prov]['feat']}")
            
            fig_imp, ax_imp = plt.subplots(figsize=(8, 3))
            sns.barplot(x=clf.feature_importances_[::-1], 
                        y=[fix_text('المخاطرة'), fix_text('رأس المال'), fix_text('القطاع'), fix_text('المحافظة')], 
                        palette="Blues_r")
            st.pyplot(fig_imp)

with tabs[1]:
    st.header("📍 تحليل المقارنة المكانية")
    comparison = []
    for p in ALL_PROVINCES:
        inf = (live_rate / 11000) * PROVINCE_INFO[p]['cost']
        res_p = le_t.inverse_transform(clf.predict([[le_p.transform([p])[0], le_s.transform([u_sect])[0], (u_budget/1000)/inf, u_risk]]))[0]
        comparison.append({"المحافظة": p, "القرار": res_p, "السمة": PROVINCE_INFO[p]['feat']})
    st.table(pd.DataFrame(comparison))

with tabs[2]:
    st.header("💡 مبتكر الفرص الذكي")
    u_budget_inv = st.number_input("اختبر ميزانية مختلفة ($)", value=u_budget, key="inv_b")
    if st.button("كشف أفضل الفرص"):
        opps = []
        for p in ALL_PROVINCES:
            for s in ALL_SECTORS:
                inf = (live_rate / 11000) * PROVINCE_INFO[p]['cost']
                # الفهرس 2 يمثل احتمال "فرصة واعدة"
                prob = clf.predict_proba([[le_p.transform([p])[0], le_s.transform([s])[0], (u_budget_inv/1000)/inf, u_risk]])[0]
                opps.append({'p': p, 's': s, 'score': prob[2], 'res': le_t.inverse_transform([clf.predict([[le_p.transform([p])[0], le_s.transform([s])[0], (u_budget_inv/1000)/inf, u_risk]])[0]])[0]})
        top = pd.DataFrame(opps).sort_values(by='score', ascending=False).head(6)
        cols = st.columns(3)
        for i, (idx, row) in enumerate(top.iterrows()):
            with cols[i % 3]: st.success(f"📍 **{row['p']}**\n\n🏗️ **{row['s']}**\n\n⭐ {row['res']}")

with tabs[3]:
    st.header("📊 مخرجات التحقق العلمي")
    m1, m2 = st.columns(2)
    m1.metric("دقة التنبؤ", f"{accuracy:.2%}")
    m2.metric("ثبات النموذج", f"{cv_results.mean():.2%}")
    fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Risk', 'Medium', 'Promise'], 
                yticklabels=['Risk', 'Medium', 'Promise'])
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    st.pyplot(fig_cm)