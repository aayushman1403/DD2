import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the model object
with open('MilkQuality.pkl', 'rb') as f:
    model = pickle.load(f)

quality_labels = {2: "good", 1: "medium", 0: "bad"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    ph = float(request.form.get('ph'))
    taste = float(request.form.get('taste'))
    odor = float(request.form.get('odor'))
    fat = float(request.form.get('fat'))
    turb = float(request.form.get('turbidity'))
    color = float(request.form.get('color'))

    # Retrieve the value of Temperature field
    temp = request.form.get('temperature')

    # Check if the Temperature field is not empty
    if temp is not None:
        # Convert temperature to float if not empty
        temp = float(temp)
    else:
        # Handle the case where temperature field is empty
        return render_template('index.html', output='Please enter a value for Temperature')

    input_data = [ph, temp, taste, odor, fat, turb, color]
    final_input = [np.array(input_data)]
    prediction = model.predict(final_input)
    quality_label = quality_labels.get(prediction[0], "unknown")

    return render_template('index.html', output=f'The quality of the milk is {quality_label}')

if __name__ == '__main__':
    app.run()
