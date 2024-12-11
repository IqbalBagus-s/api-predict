from dotenv import load_dotenv
import os
import mysql.connector
from datetime import datetime

# Konfigurasi koneksi database
# Load konfigurasi dari .env
load_dotenv()

db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

# Fungsi untuk menyimpan prediksi ke database
def save_prediction(name, input_data, description):
    try:
        db_data = {
            "name": name,
            "age": input_data["age"],
            "hypertension": input_data["hypertension"],
            "heart_disease": input_data["heart_disease"],
            "body_mass_index": input_data["bmi"],
            "HbA1c_level": input_data["HbA1c_level"],
            "blood_glucose_level": input_data["blood_glucose_level"],
            "gender": input_data["gender_encoded"],
            "smoking_history": input_data["smoking_history_encoded"],
            "diabetes_category": description,
            "check_date": datetime.now()
        }

        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        query = """
            INSERT INTO histories (
                name, age, hypertension, heart_disease, 
                body_mass_index, HbA1c_level, blood_glucose_level, 
                gender, smoking_history, diabetes_category, check_date
            ) VALUES (
                %(name)s, %(age)s, %(hypertension)s, %(heart_disease)s, 
                %(body_mass_index)s, %(HbA1c_level)s, %(blood_glucose_level)s, 
                %(gender)s, %(smoking_history)s, %(diabetes_category)s, %(check_date)s
            )
        """
        cursor.execute(query, db_data)
        connection.commit()

        print("Prediction saved successfully.")
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
