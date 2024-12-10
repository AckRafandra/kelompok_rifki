from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    """Menampilkan halaman utama dengan form prediksi."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Menerima data dari frontend dan memberikan hasil prediksi."""
    try:
        data = request.get_json()
        studytime = int(data.get('studytime'))
        absences = int(data.get('absences'))
        g1 = int(data.get('G1'))
        g2 = int(data.get('G2'))

        if not (1 <= studytime <= 4):
            return jsonify({'error': 'Studytime harus antara 1-4'}), 400
        if not (0 <= absences <= 93):
            return jsonify({'error': 'Absences harus antara 0-93'}), 400
        if not (0 <= g1 <= 20):
            return jsonify({'error': 'G1 harus antara 0-20'}), 400
        if not (0 <= g2 <= 20):
            return jsonify({'error': 'G2 harus antara 0-20'}), 400

        # Membuat dataframe untuk prediksi
        user_data = pd.DataFrame({
            'studytime': [studytime],
            'absences': [absences],
            'G1': [g1],
            'G2': [g2]
        })

        # Scale user data
        user_data_scaled = scaler.transform(user_data)

        # Prediksi nilai G3
        predicted_g3 = model.predict(user_data)[0]

        # Tentukan hasil prediksi (Pass atau Fail)
        result = "Pass" if predicted_g3 >= 15 else "Fail"

        return jsonify({
            'predicted_G3': predicted_g3,
            'result': result
        })

    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
