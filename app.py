
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Ticari Alacak Risk Değerlendirme Sistemi",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Ticari Alacak Yönetimi & Risk Değerlendirme Dashboard'u")
st.markdown("---")

@st.cache_resource
def load_models():
    class_model = joblib.load('xgb_classifier_model.pkl')
    reg_model = joblib.load('xgb_regressor_model.pkl')
    features = joblib.load('model_features.pkl')
    return class_model, reg_model, features

try:
    xgb_class, xgb_reg, model_features = load_models()
    st.sidebar.success("✅ Modeller Başarıyla Yüklendi")
except Exception as e:
    st.sidebar.error("❌ Modeller Yüklenemedi.")
    st.stop()

# Form Yapısı (Hata almamak ve kontrollü çalıştırmak için)
st.sidebar.header("🔍 Yeni Fatura Parametreleri")

with st.sidebar.form("risk_form"):
    payment_method = st.number_input("Ödeme Yöntemi Kodu (Payment_Method)", min_value=0, max_value=20, value=1)
    rank_order = st.number_input("Müşterinin Sipariş Sıralaması (Rank_of_order)", min_value=1, max_value=500, value=10)
    doc_month = st.slider("Fatura Kesim Ayı (Doc_Month)", 1, 12, 5)
    doc_year = st.selectbox("Fatura Kesim Yılı (Doc_Year)", [2023, 2024, 2025, 2026])
    amount = st.number_input("Fatura Tutarı (TL)", min_value=100.0, max_value=1000000.0, value=25000.0)
    zipcode = st.number_input("Posta Kodu (Zipcode)", min_value=10000, max_value=99999, value=34000)
    
    submitted = st.form_submit_button("🚀 Risk Skorunu Hesapla")

if submitted:
    input_data = pd.DataFrame(0, index=[0], columns=model_features)
    
    if 'Payment_Method_description' in input_data.columns: input_data['Payment_Method_description'] = payment_method
    if 'Rank_of_order_by_customer' in input_data.columns: input_data['Rank_of_order_by_customer'] = rank_order
    if 'Doc_Month' in input_data.columns: input_data['Doc_Month'] = doc_month
    if 'Doc_Year' in input_data.columns: input_data['Doc_Year'] = doc_year
    if 'Amount' in input_data.columns: input_data['Amount'] = amount
    if 'Zipcode' in input_data.columns: input_data['Zipcode'] = zipcode

    prob_delay = xgb_class.predict_proba(input_data)[0][1] * 100
    pred_days = max(0, xgb_reg.predict(input_data)[0])

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Gecikme Olasılığı (%)", value=f"%{prob_delay:.1f}")
    with col2:
        st.metric(label="Tahmini Gecikme Süresi", value=f"{pred_days:.1f} Gün")
    with col3:
        if prob_delay >= 70 or pred_days > 30:
            st.error("🔴 Yüksek Risk Grubu")
        elif prob_delay >= 30 or pred_days > 7:
            st.warning("🟡 Orta Risk Grubu")
        else:
            st.success("🟢 Düşük Risk Grubu")

    st.markdown("---")
    st.subheader("💡 Operasyonel Tahsilat & Risk Aksiyon Önerisi")
    if prob_delay >= 70 or pred_days > 30:
        st.write("⚠️ **Aksiyon:** Fatura vadesinden 5 gün önce erken hatırlatma araması yapılmalı. Müşteri limitleri gözden geçirilmeli.")
    elif prob_delay >= 30 or pred_days > 7:
        st.write("🔔 **Aksiyon:** Otomatik e-posta/SMS uyarısı planlanmalı.")
    else:
        st.write("✅ **Aksiyon:** Standart süreç takibi yeterlidir.")
