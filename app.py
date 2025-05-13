from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model from the pickle file
with open('Grip.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all 22 input features from the form
        inputs = [
            float(request.form['Current_J0']),
            float(request.form['Temperature_T0']),
            float(request.form['Current_J1']),
            float(request.form['Temperature_J1']),
            float(request.form['Current_J2']),
            float(request.form['Temperature_J2']),
            float(request.form['Current_J3']),
            float(request.form['Temperature_J3']),
            float(request.form['Current_J4']),
            float(request.form['Temperature_J4']),
            float(request.form['Current_J5']),
            float(request.form['Temperature_J5']),
            float(request.form['Speed_J0']),
            float(request.form['Speed_J1']),
            float(request.form['Speed_J2']),
            float(request.form['Speed_J3']),
            float(request.form['Speed_J4']),
            float(request.form['Speed_J5']),
            float(request.form['Tool_current']),
            float(request.form['cycle']),
        ]

        input_data = pd.DataFrame([inputs], columns=[
            'Current_J0', 'Temperature_T0', 'Current_J1', 'Temperature_J1',
            'Current_J2', 'Temperature_J2', 'Current_J3', 'Temperature_J3',
            'Current_J4', 'Temperature_J4', 'Current_J5', 'Temperature_J5',
            'Speed_J0', 'Speed_J1', 'Speed_J2', 'Speed_J3', 'Speed_J4', 'Speed_J5',
            'Tool_current', 'cycle '
        ])

        prediction = model.predict(input_data)[0]

        # Check if prediction is 0 and set appropriate message
        prediction_text = "No Grip Lost" if prediction == 0 else f"Grip Lost: {prediction}"

        return render_template('home.html', prediction_text=prediction_text)
    except Exception as e:
        return render_template('home.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)