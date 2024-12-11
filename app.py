from flask import Flask
from routes.diabetes_routes import diabetes_bp
from controllers.diabetes_controller import load_model
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Load model saat server dimulai
load_model()

app = Flask(__name__)
app.register_blueprint(diabetes_bp, url_prefix='/api')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
