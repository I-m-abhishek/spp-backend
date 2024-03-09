import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # Import joblib for saving/loading scalers

# Load the dataset
dts = pd.read_csv('solarpowergeneration.csv')

# Extract features and target variable
X = dts.iloc[:, :-1].values
y = dts.iloc[:, -1].values
y = np.reshape(y, (-1, 1))

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Ensure y_train and y_test are arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# Feature Scaling
sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)

sc_y = StandardScaler()
y_train_scaled = sc_y.fit_transform(y_train)
y_test_scaled = sc_y.transform(y_test)

# Save the trained scalers using joblib
joblib.dump(sc_X, 'scaler-x.pkl')
joblib.dump(sc_y, 'scaler-y.pkl')

# Creating Neural Network Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', kernel_initializer='normal', input_dim=X_train_scaled.shape[1]),
    tf.keras.layers.Dense(64, activation='relu', kernel_initializer='normal'),
    tf.keras.layers.Dense(1)
])
model.compile(loss='mse', optimizer='adam', metrics=['RootMeanSquaredError'])

# Training the Neural Network
hist = model.fit(X_train_scaled, y_train_scaled, batch_size=32, validation_data=(X_test_scaled, y_test_scaled), epochs=150, verbose=2)

# Save the trained model
model.save('solar-power-model.keras')
