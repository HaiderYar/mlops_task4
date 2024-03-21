from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Load the trained model
model = joblib.load('knn_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()

        # Convert input data to DataFrame
        df = pd.DataFrame(data, index=[0])

        # Get categorical feature names
        categorical_features = ['buying_price', 'maintenance_price', 'size_of_luggage_boot', 'safety']

        # Perform one-hot encoding for categorical features
        one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        one_hot_encoder.fit(df[categorical_features])
        X_encoded = one_hot_encoder.transform(df[categorical_features])

        # Make predictions using the loaded model
        predictions = model.predict(X_encoded)

        # Return predictions as JSON response
        return jsonify({'predictions': predictions.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
