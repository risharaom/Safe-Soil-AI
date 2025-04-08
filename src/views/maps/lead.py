from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import csv

# Load data
data = []
labels = []

with open("leadData.csv", "r") as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip header
    for row in reader:
        try:
            features = np.array(row[1:3], dtype=float)
            label = float(row[3])
            data.append(features)
            labels.append(label)
        except ValueError:
            print(f"Skipping row due to invalid data: {row}")

# Convert lists to NumPy arrays
X = np.array(data)
y = np.array(labels)

# Normalize the feature data
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_range = X_max - X_min
X_range[X_range == 0] = 1
X_norm = (X - X_min) / X_range

# Split into train/test sets
split_index = int(0.8 * len(X_norm))
X_train = X_norm[:split_index]
X_test = X_norm[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

# Linear Regression Model
class LinearRegressionModel:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def train(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        for epoch in range(self.epochs):
            predictions = np.dot(X, self.weights) + self.bias
            error = predictions - y
            
            dw = np.dot(X.T, error) / m
            db = np.mean(error)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Train the model
model = LinearRegressionModel(learning_rate=0.01, epochs=2000)
model.train(X_train, y_train)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)

        if 'latitude' not in data or 'longitude' not in data:
            return jsonify({'error': 'Latitude and Longitude are required.'}), 400

        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        print(latitude, longitude)

        user_input = np.array([[latitude, longitude]], dtype=float)
        user_input_norm = (user_input - X_min) / X_range
        prediction = model.predict(user_input_norm)[0]

        print("Prediction:", prediction)
        return jsonify({'prediction': round(prediction, 2)})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

'''from flask import Flask, request, jsonify
import numpy as np
import csv

import joblib

# Load data
data = []
labels = []

with open("leadData.csv", "r") as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip header
    for row in reader:
        try:
            features = np.array(row[1:3], dtype=float)
            label = float(row[3])
            data.append(features)
            labels.append(label)
        except ValueError:
            print(f"Skipping row due to invalid data: {row}")

# Convert lists to NumPy arrays
X = np.array(data)
y = np.array(labels)

# Normalize the feature data
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_range = X_max - X_min
X_range[X_range == 0] = 1
X_norm = (X - X_min) / X_range

# Split into train/test sets
split_index = int(0.8 * len(X_norm))
X_train = X_norm[:split_index]
X_test = X_norm[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

# Linear Regression Model
class LinearRegressionModel:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def train(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        for epoch in range(self.epochs):
            predictions = np.dot(X, self.weights) + self.bias
            error = predictions - y
            
            dw = np.dot(X.T, error) / m
            db = np.mean(error)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Train the model
model = LinearRegressionModel(learning_rate=0.01, epochs=2000)
model.train(X_train, y_train)

# Initialize Flask app
app = Flask(__name__)

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Log the received data for debugging
        print("Received data:", data)
        
        if 'latitude' not in data or 'longitude' not in data:
            return jsonify({'error': 'Latitude and Longitude are required.'}), 400
        
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        print(latitude,longitude)
        
        user_input = np.array([[latitude, longitude]], dtype=float)
        user_input_norm = (user_input - X_min) / X_range

        prediction = model.predict(user_input_norm)[0]
        
        # Log the prediction for debugging
        print("Prediction:", prediction)

        return jsonify({'prediction': round(prediction, 2)})
    
    except Exception as e:
        # Log the error and send back a meaningful response
        print("Error:", e)
        return jsonify({'error': str(e)}), 500


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
'''
'''
# Save model and preprocessing info
joblib.dump(model, 'linear_model.pkl')
joblib.dump({'X_min': X_min, 'X_range': X_range}, 'preprocessing_params.pkl')
'''


'''
from flask import Flask, request, jsonify
import numpy as np
import csv
import joblib

# Load data
data = []
labels = []

with open("leadData.csv", "r") as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip header
    for row in reader:
        try:
            features = np.array(row[1:3], dtype=float)
            label = float(row[3])
            data.append(features)
            labels.append(label)
        except ValueError:
            print(f"Skipping row due to invalid data: {row}")

# Convert lists to NumPy arrays
X = np.array(data)
y = np.array(labels)

# Normalize the feature data
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_range = X_max - X_min
X_range[X_range == 0] = 1
X_norm = (X - X_min) / X_range

# Split into train/test sets
split_index = int(0.8 * len(X_norm))
X_train = X_norm[:split_index]
X_test = X_norm[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

# Linear Regression Model
class LinearRegressionModel:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def train(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        for epoch in range(self.epochs):
            predictions = np.dot(X, self.weights) + self.bias
            error = predictions - y
            
            dw = np.dot(X.T, error) / m
            db = np.mean(error)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Train the model
model = LinearRegressionModel(learning_rate=0.01, epochs=2000)
model.train(X_train, y_train)

# Initialize Flask app
app = Flask(__name__)

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if 'latitude' not in data or 'longitude' not in data:
            return jsonify({'error': 'Latitude and Longitude are required.'}), 400

        latitude = float(data['latitude'])
        longitude = float(data['longitude'])

        user_input = np.array([[latitude, longitude]], dtype=float)
        user_input_norm = (user_input - X_min) / X_range

        prediction = model.predict(user_input_norm)[0]

        return jsonify({'prediction': round(prediction, 2)})
    
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500
# Run the app
if __name__ == "__main__":
    app.run(debug=True)



# Save model and preprocessing info
joblib.dump(model, 'linear_model.pkl')
joblib.dump({'X_min': X_min, 'X_range': X_range}, 'preprocessing_params.pkl')
'''