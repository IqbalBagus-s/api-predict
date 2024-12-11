from flask import Blueprint, request, jsonify
from controllers.diabetes_controller import predict_diabetes, save_prediction

diabetes_bp = Blueprint('diabetes', __name__)

@diabetes_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        name = data.get("name")  # Ambil nama pengguna dari input

        if not name:
            return jsonify({"error": "Name is required"}), 400

        result = predict_diabetes(data)

        # Simpan hasil prediksi ke database
        save_prediction(name, result["input"], result["description"])

        # Kembalikan respons tanpa parsedInput dan predictedClass
        return jsonify({
            "input": result["input"],
            "description": result["description"]
        })
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
