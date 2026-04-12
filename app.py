import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import os

# Konfigurasi Halaman Dasar (Layout Centered)
st.set_page_config(page_title="Deteksi Depresi Mahasiswa", page_icon="🧠", layout="centered")

# Load Model, Scaler, dan Fitur
@st.cache_resource
def load_assets():
    model_dir = 'model'
    model = joblib.load(os.path.join(model_dir, 'model_depresi_fasttrack.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler_fasttrack.pkl'))
    fitur_pilihan = joblib.load(os.path.join(model_dir, 'fitur_pilihan.pkl'))
    return model, scaler, fitur_pilihan

model, scaler, fitur_pilihan = load_assets()

# HEADER UTAMA
st.markdown("<h1 style='text-align: center;'>🧠 Deteksi Dini Depresi Mahasiswa</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Aplikasi ini dikembangkan untuk memberikan kesadaran awal terkait kesehatan mental mahasiswa menggunakan model yang telah dilatih dengan data akademik dan gaya hidup.</p>", unsafe_allow_html=True)

st.markdown("---")

# SEKSI 1: PANDUAN FITUR
st.markdown("### 📖 Panduan Fitur")
st.markdown("""
1. **Performa Akademik (IPK):** Nilai akademik Anda (Skala 0-10)
2. **Tingkat Kepuasan Belajar:** Rasa bahagia dalam proses kuliah (0-5)
3. **Tingkat Tekanan Belajar:** Seberapa berat beban kuliah Anda (0-5)
4. **Tingkat Stres Finansial:** Kekhawatiran Anda terkait biaya (1-5)
""")

st.markdown("---")

# SEKSI 2: FORMULIR INPUT
st.subheader("📝 Formulir Assessment")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia Anda", min_value=15, max_value=40, value=20)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    cgpa = st.slider("1. Performa Akademik (IPK 0-10)", 0.0, 10.0, 8.0, step=0.1)
    study_satisfaction = st.select_slider("2. Tingkat Kepuasan Belajar", options=list(range(6)), value=3)
    work_study_hours = st.number_input("Jam Belajar/Kerja per Hari", 0, 24, 8)

with col2:
    academic_pressure = st.select_slider("3. Tingkat Tekanan Belajar", options=list(range(6)), value=3)
    financial_stress = st.select_slider("4. Tingkat Stres Finansial", options=list(range(1, 6)), value=3)
    dietary_habits = st.selectbox("Pola Makan Harian", ["Healthy", "Unhealthy", "Others"])
    suicidal_thoughts = st.radio("Pernah terpikir menyakiti diri sendiri?", ["No", "Yes"], horizontal=True)

# PROSES PREDIKSI
if st.button("🚀 Jalankan Analisis", use_container_width=True):
    # Mapping data
    is_male = 1 if gender == "Male" else 0
    is_diet_healthy = 1 if dietary_habits == "Healthy" else 0
    is_diet_unhealthy = 1 if dietary_habits == "Unhealthy" else 0
    is_suicidal_yes = 1 if suicidal_thoughts == "Yes" else 0

    # Dataframe untuk model
    input_data = pd.DataFrame([[
        is_suicidal_yes, academic_pressure, financial_stress, age, cgpa, 
        work_study_hours, study_satisfaction, is_diet_unhealthy, is_male, is_diet_healthy
    ]], columns=fitur_pilihan)

    # Scaling & Prediksi
    input_scaled = scaler.transform(input_data)
    hasil_prediksi = model.predict(input_scaled)
    probabilitas = model.predict_proba(input_scaled)[0][1] * 100

    st.markdown("---")
    
    # SEKSI 3: VISUALISASI HASIL
    st.subheader("📊 Hasil Analisis Psikologis")
    
    v_col1, v_col2 = st.columns([1.2, 1])
    
    with v_col1:
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probabilitas,
            title = {'text': "Tingkat Risiko Indikasi", 'font': {'size': 18}},
            number = {'suffix': "%", 'font': {'size': 35}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "black", 'thickness': 0.2},
                'steps': [
                    {'range': [0, 40], 'color': "#00CC96"},
                    {'range': [40, 70], 'color': "#FFC107"},
                    {'range': [70, 100], 'color': "#EF553B"}
                ]
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with v_col2:
        # Bar Chart Parameter
        labels = ['Tekanan', 'Finansial', 'Kepuasan']
        values = [academic_pressure, financial_stress, study_satisfaction]
        fig_bar = go.Figure(data=[go.Bar(x=labels, y=values, marker_color='#636EFA')])
        fig_bar.update_layout(title_text='Parameter Psikologis', height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    # SEKSI 4: KESIMPULAN AKHIR
    if hasil_prediksi[0] == 1:
        st.error(f"🚨 **STATUS: PERLU PERHATIAN (Risiko: {probabilitas:.1f}%)**")
        st.info("**Rekomendasi:** Pola data menunjukkan beban pikiran yang cukup berat. Cobalah beristirahat sejenak atau bercerita kepada orang terdekat/konselor kampus.")
    else:
        st.success(f"✅ **STATUS: AMAN TERKENDALI (Risiko: {probabilitas:.1f}%)**")
        st.info("**Rekomendasi:** Kondisi mental Anda saat ini terlihat stabil. Tetap jaga keseimbangan waktu antara belajar dan waktu luang!")

# Footer
st.divider()
st.markdown("<p style='text-align: center; color: #808495; font-size: 0.9em;'>Suluh Yoga Pratama | Bengkel Koding</p>", unsafe_allow_html=True)