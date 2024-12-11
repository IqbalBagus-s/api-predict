from flask import Flask
from routes.diabetes_routes import diabetes_bp
from controllers.diabetes_controller import load_model

app = Flask(__name__)
app.register_blueprint(diabetes_bp, url_prefix='/api')

# Load model saat server dimulai
load_model()

if __name__ == '__main__':
    app.run(debug=True, port=5000)