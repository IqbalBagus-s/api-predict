import tensorflow as tf
from google.cloud import storage
import numpy as np
from models.model_diabetes import save_prediction
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Konfigurasi untuk menyimpan model lokal sementara
MODEL_PATH = './models/diabetes_model2.keras'
BUCKET_NAME = 'bucketgluco'  # Ganti dengan nama bucket Anda
MODEL_FILE_PATH = 'models/diabetes_model2.keras'  # Path di dalam bucket
model = None  # Deklarasikan variabel model global

# Load the model
def load_model():
    global model  # Pastikan variabel global digunakan
    try:
        # Periksa apakah file model sudah ada di direktori lokal
        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found locally at {MODEL_PATH}. Downloading from GCS...")
            download_model_from_gcs()  # Unduh model dari GCS jika tidak ada

        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)  # Muat model dari file lokal
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error during model loading: {e}")


# Fungsi untuk mengunduh model dari Google Cloud Storage
def download_model_from_gcs():
    import tensorflow as tf
    from google.cloud import storage

    client = storage.Client()
    bucket_name = "bucketgluco"
    blob_name = "models/diabetes_model2.keras"
    local_model_path = MODEL_PATH

    try:
        print(f"Downloading model from GCS: {bucket_name}/{blob_name} to {local_model_path}...")
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_model_path)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model from GCS: {e}")
        raise


# Map textual inputs to numeric values
def parse_input(data):
    # Mapping for categorical inputs
    yes_no_map = {"yes": 1, "no": 0}
    gender_map = {"male": 1, "female": 0}
    smoking_map = {
        "No Info": 0,
        "never": 1,
        "former": 2,
        "current": 3,
        "not current": 4,
        "ever": 5,
    }

    try:
        # Convert textual inputs to numeric values
        parsed_data = {
            "age": float(data["age"]),
            "hypertension": yes_no_map.get(data["hypertension"].lower(), 0),
            "heart_disease": yes_no_map.get(data["heart_disease"].lower(), 0),
            "bmi": float(data["bmi"]),
            "HbA1c_level": float(data["HbA1c_level"]),
            "blood_glucose_level": float(data["blood_glucose_level"]),
            "gender_encoded": gender_map.get(data["gender_encoded"].lower(), 0),
            "smoking_history_encoded": smoking_map.get(data["smoking_history_encoded"], 0),
        }
        return parsed_data
    except Exception as e:
        raise ValueError(f"Error parsing input data: {e}")

# Predict function
def predict_diabetes(data):
    global model  # Pastikan mengakses model global
    if not model:
        raise RuntimeError("Model not loaded")

    # Parse and validate input
    try:
        parsed_data = parse_input(data)
    except Exception as e:
        return {"error": f"Error parsing input data: {str(e)}"}

    # Convert parsed data into array
    features = [
        "age", "hypertension", "heart_disease", "bmi",
        "HbA1c_level", "blood_glucose_level", "gender_encoded", "smoking_history_encoded"
    ]
    input_data = [parsed_data[feature] for feature in features]
    input_array = np.array([input_data], dtype=np.float32)

    try:
        # Make prediction
        predictions = model.predict(input_array)
        predicted_class = np.argmax(predictions[0])

        # Define result descriptions
        results = [
            "Normal",
            "Pra-diabetes",
            "Controlled Diabetes",
            "Uncontrolled Diabetes",
        ]

        return {
            "input": data,
            "description": results[predicted_class]
        }
    except Exception as e:
        return {"error": f"Error making prediction: {str(e)}"}
