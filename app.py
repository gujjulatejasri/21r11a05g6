import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, session
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Secret key for session management (you can change this to a secure value)
app.secret_key = 'your_secret_key'

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Data file for storing user input
data_file = 'user_data.csv'

# Try to load existing user data, if not create a new DataFrame
try:
    df = pd.read_csv(data_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=["Age", "Maternal Health Condition", "Child Birth Weight", "Number of Previous Children"])

# Mapping of health conditions to numeric values
health_condition_map = {
    'healthy': 0,
    'high blood pressure': 1,
    'diabetes': 2,
    'anemia': 3,
}

# Function to retrain the model based on updated data
def retrain_model():
    global model
    try:
        # Load the stored data
        df = pd.read_csv(data_file)

        # Prepare the data for training
        X = df[["Age", "Maternal Health Condition", "Child Birth Weight", "Number of Previous Children"]]
        y = df["Mortality Risk"]  # Assuming you have a column that labels mortality risk (1 for risk, 0 for no risk)

        # Convert health condition to numeric
        X["Maternal Health Condition"] = X["Maternal Health Condition"].map(health_condition_map)

        # Train the model
        model = RandomForestClassifier()  # Use the desired model
        model.fit(X, y)

        # Save the trained model
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        print(f"Error retraining model: {e}")
        return False

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Simple hardcoded login (username: admin, password: password)
        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            return redirect(url_for('predict'))  # Redirect to prediction page after login
        else:
            return "Invalid credentials, please try again."

    return render_template('login.html')

# Route to logout
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# Index route (Home page)
@app.route('/')
def index():
    # Check if user is logged in
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    return render_template('index.html')

# Route for entering data
@app.route('/enter_data', methods=['POST'])
def enter_data():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    global df
    if request.method == 'POST':
        try:
            # Collect data from form
            age = float(request.form['Mother\'s Age'])
            health_condition = request.form['Maternal Health Condition'].strip().lower()
            child_birth_weight = float(request.form['Child Birth Weight'])
            previous_children = int(request.form['Number of Previous Children'])

            # Convert health condition to numeric
            health_condition_numeric = health_condition_map.get(health_condition)
            if health_condition_numeric is None:
                return "Invalid maternal health condition. Please check your input."

            # Create a DataFrame for new data and append it to existing data
            new_data = pd.DataFrame({
                "Age": [age],
                "Maternal Health Condition": [health_condition],
                "Child Birth Weight": [child_birth_weight],
                "Number of Previous Children": [previous_children]
            })

            df = pd.concat([df, new_data], ignore_index=True)
            df.to_csv(data_file, index=False)

            # Retrain model with updated data
            if retrain_model():
                return render_template('index.html', message="Data stored and model retrained successfully. Now you can predict mortality.")
            else:
                return render_template('index.html', message="Data stored, but model retraining failed.")

        except Exception as e:
            return f"An error occurred: {e}"

# Route for prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            # Collect data from form
            age = float(request.form['Mother\'s Age'])
            health_condition = request.form['Maternal Health Condition'].strip().lower()
            child_birth_weight = float(request.form['Child Birth Weight'])
            previous_children = int(request.form['Number of Previous Children'])

            # Convert health condition to numeric
            health_condition_numeric = health_condition_map.get(health_condition)
            if health_condition_numeric is None:
                return "Invalid maternal health condition. Please check your input."

            # Prepare features for prediction
            features = np.array([[age, health_condition_numeric, child_birth_weight, previous_children]])
            features_scaled = scaler.transform(features)  # If scaling is applied, otherwise skip this line
            prediction = model.predict(features_scaled)
            result = prediction[0]

            # Determine the result and assign CSS class
            if result == 'Yes':
                result = 'Risk of Mortality'
                result_class = 'risk'  # CSS class for risk
            else:
                result = 'No Risk of Mortality'
                result_class = 'no-risk'  # CSS class for no risk

            return render_template('index.html', prediction_text=result, result_class=result_class)

        except Exception as e:
            return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
