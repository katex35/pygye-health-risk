from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import ssl

# disable SSL verification for development
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# load the data and train the model
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train.values.ravel())

@app.route('/')
def home():
    # Serve the frontend HTML file
    return send_file('index.html')

@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "version": "1.0.0"
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # check if all required fields are present
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "error": f"Missing required field: {field}",
                    "status": "error"
                }), 400

        # convert the input data to numpy array
        input_data = np.array([[
            data['age'],
            data['sex'],
            data['cp'],
            data['trestbps'],
            data['chol'],
            data['fbs'],
            data['restecg'],
            data['thalach'],
            data['exang'],
            data['oldpeak'],
            data['slope'],
            data['ca'],
            data['thal']
        ]])

        # scale the input data
        input_scaled = scaler.transform(input_data)
        
        # predict the class and probability
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

        return jsonify({
            "prediction": int(prediction[0]),
            "probability": float(probability[0][1]),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    return jsonify({
        "features": {
            "age": "Age in years",
            "sex": "Gender (1 = male, 0 = female)",
            "cp": "Chest pain type (0-3)",
            "trestbps": "Resting blood pressure in mm Hg",
            "chol": "Serum cholesterol in mg/dl",
            "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
            "restecg": "Resting electrocardiographic results (0-2)",
            "thalach": "Maximum heart rate achieved",
            "exang": "Exercise induced angina (1 = yes, 0 = no)",
            "oldpeak": "ST depression induced by exercise relative to rest",
            "slope": "Slope of the peak exercise ST segment (0-2)",
            "ca": "Number of major vessels colored by fluoroscopy (0-3)",
            "thal": "Thalassemia test result (0-3)"
        },
        "status": "success"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000) 