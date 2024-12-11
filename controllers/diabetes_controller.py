import tensorflow as tf
import numpy as np
from models.model_diabetes import save_prediction
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the model path from environment variables
MODEL_PATH = os.getenv("MODEL_PATH")
model = None

# Load the model
def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

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
    if not model:
        raise RuntimeError("Model not loaded")

    # Parse and validate input
    parsed_data = parse_input(data)

    # Convert parsed data into array
    features = [
        "age", "hypertension", "heart_disease", "bmi",
        "HbA1c_level", "blood_glucose_level", "gender_encoded", "smoking_history_encoded"
    ]
    input_data = [parsed_data[feature] for feature in features]
    input_array = np.array([input_data], dtype=np.float32)

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
