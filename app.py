from flask import Flask
from routes.diabetes_routes import diabetes_bp
from controllers.diabetes_controller import load_model
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.register_blueprint(diabetes_bp, url_prefix='/api')

# Load model saat server dimulai
load_model()

if __name__ == '__main__':
    # Gunakan variabel environment untuk konfigurasi
    debug_mode = os.getenv("DEBUG", "False").lower() == "true"
    port = int(os.getenv("PORT", 5000))
    
    # Jalankan aplikasi pada host 0.0.0.0 agar dapat diakses dari luar
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
