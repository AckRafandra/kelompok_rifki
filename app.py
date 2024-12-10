import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model dan scaler
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Custom CSS
st.markdown(
    """
    <style>
    body {
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        color: #333;
        background: #f5f5f5;
    }
    h1 {
        max-width: 400px;
        margin: 20px auto;
        padding: 20px;
        background: white;
        border-radius: 8px;
        text-align: center;
        margin-top: 20px;
        color: #000000;
    }
    .form-container {
        max-width: 400px;
        margin: 20px auto;
        padding: 20px;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        position: relative;
    }
    .stButton button {
        width: 100%;
        padding: 10px;
        background-color: #1769ce;
        color: white;
        border: none;
        border-radius: 4px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #0b5ba1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown("<h1>Prediksi Nilai Akhir Siswa</h1>", unsafe_allow_html=True)

# Input Form
with st.form(key="prediction_form"):
    st.markdown("<div class='form-container'>", unsafe_allow_html=True)
    studytime = st.number_input("Waktu Belajar (1-4):", min_value=1, max_value=4, step=1)
    absences = st.number_input("Jumlah Ketidakhadiran (0-93):", min_value=0, max_value=93, step=1)
    g1 = st.number_input("Nilai G1 (0-20):", min_value=0, max_value=20, step=1)
    g2 = st.number_input("Nilai G2 (0-20):", min_value=0, max_value=20, step=1)
    submit_button = st.form_submit_button("Prediksi")
    st.markdown("</div>", unsafe_allow_html=True)

# Prediction Logic
if submit_button:
    try:
        # Data untuk prediksi
        user_data = pd.DataFrame({
            'studytime': [studytime],
            'absences': [absences],
            'G1': [g1],
            'G2': [g2]
        })

        # Normalisasi dengan scaler
        user_data_scaled = scaler.transform(user_data)

        # Prediksi nilai G3
        predicted_g3 = model.predict(user_data_scaled)[0]

        # Hasil prediksi
        st.markdown("<h2>Hasil Prediksi</h2>", unsafe_allow_html=True)
        st.success(f"**Prediksi Nilai Akhir (G3):** {round(predicted_g3, 2)}")

        # Tentukan kelulusan
        if predicted_g3 >= 15:
            st.success("🎉 **Prediksi: LULUS (Pass)**")
        else:
            st.error("😞 **Prediksi: TIDAK LULUS (Fail)**")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
