# server.py
from flask import Flask, request, jsonify
from predictmodel import predict
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
@app.route('/predict', methods=['POST'])
def make_prediction():
    try:
        data = request.json
        input_data = data['input_data']
        
        # Call the predict function from predict_model.py
        prediction = predict(input_data)

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/')
def home():
    return "Hello Home"


if __name__ == '__main__':
    app.run(host="0.0.0.0")
