# predict_model.py
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

def predict(input_data):
    try:
        # Load the trained model
        loaded_model = tf.keras.models.load_model('solar-power-model.h5')

        # Load the StandardScaler used during training
        sc_X = StandardScaler()
        sc_y = StandardScaler()

        # Load the scalers from a file or another saved source
        sc_X_filename = 'scaler-x.pkl'
        sc_y_filename = 'scaler-y.pkl'

        sc_X = joblib.load(sc_X_filename)
        sc_y = joblib.load(sc_y_filename)
    
        # Feature Scaling for new input data
        new_input_data_scaled = sc_X.transform(np.array(input_data).reshape(1, -1))

        # Predict using the loaded model
        predicted_scaled = loaded_model.predict(new_input_data_scaled)

        # Inverse transform to get the original scale
        predicted_original = sc_y.inverse_transform(predicted_scaled)
        # print(predicted_original)
        return predicted_original.tolist()

    except Exception as e:
        return str(e)
    

# new_input_data = [2.17, 31, 1035, 0, 0, 0, 0, 0, 0, 0, 6.37, 312.71, 9.36, 22.62, 6.62, 337.62, 24.48, 58.753108, 83.237322, 128.33543]

# Example usage:
# prediction = predict(new_input_data)
# print("Predicted Solar Power Generation:", prediction)
# predict(new_input_data)