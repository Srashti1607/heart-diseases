from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import time

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    start_time = time.time()
    # Extract data from form
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(final_features)
   

    # Make prediction
    prediction = model.predict(scaled_features)
    output = 'healthy' if prediction[0] == 1 else 'Unhealthy'

    end_time = time.time()

    time_taken = end_time - start_time

    prediction_text = f"Prediction: {output}. Time taken: {time_taken:.2f} seconds"

    return render_template('index.html', prediction_text=prediction_text)

    # return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)