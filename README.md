# GlucoGuide API

GlucoGuide API adalah aplikasi berbasis Flask yang menggunakan model TensorFlow untuk memprediksi tingkat resiko diabetes berdasarkan input pengguna. API ini menerima data kesehatan pengguna melalui endpoint dan memberikan klasifikasi apakah seseorang berisiko terkena diabetes atau tidak.

## Fitur

- **Input**: Pengguna dapat mengirimkan data berupa fitur medis seperti usia, BMI, HbA1c level, dan lain-lain.
- **Prediksi**: API akan mengembalikan prediksi tentang status diabetes pengguna (misalnya, "uncontrolled diabetes", "at risk", dll).
- **Model**: Model TensorFlow digunakan untuk memproses input dan memberikan hasil prediksi.

## Instalasi

1. **Clone Repository**  
   Clone repo ini ke mesin lokal Anda:
   ```
   git clone <URL_REPO>
   cd <FOLDER_PROJECT>
2. **Install Dependency**
   Instal dependensi yang dibutuhkan menggunakan pip:
   ```
   pip install -r requirements.txt

## Menjalankan Aplikasi
Untuk menjalankan aplikasi, cukup jalankan file app.py:
```
python app.py
```
Setelah server berjalan, Anda akan melihat output seperti ini di terminal:
```
Running on http://127.0.0.1:3000/ (Press CTRL+C to quit)
```
## Endpoint API
**URL**: http://127.0.0.1:3000/api/predict

**Metode**: **POST**

**Body Request (JSON)**: Kirimkan data JSON berisi informasi medis pengguna untuk prediksi. Berikut adalah contoh data yang bisa dikirimkan:
```
{
    "age": 45,
    "hypertension": 1,
    "heart_disease": 0,
    "bmi": 28.5,
    "HbA1c_level": 6.0,
    "blood_glucose_level": 120,
    "gender_encoded": 1,
    "smoking_history_encoded": 0
}
```
**Response**: API akan mengembalikan hasil prediksi dalam format JSON. Contoh response:
```
{
    "prediction": "uncontrolled diabetes"
}
```
## Catatan
- Pastikan Anda sudah menginstal semua dependensi dengan benar dan server Flask berjalan sebelum menguji API.
- Jika Anda melihat error terkait TensorFlow atau model, pastikan bahwa model .keras sudah ada di folder yang tepat dan kompatibel dengan versi TensorFlow yang digunakan.
