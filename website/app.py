from flask import Flask, request, render_template, redirect, url_for
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
with open('model/predictor.pickle', 'rb') as file:
    model = pickle.load(file)

# Define mapping for categorical variables
# These mappings should match the encoding used during training
gender_mapping = {'Male': 0, 'Female': 1, 'Non-binary': 2}
education_mapping = {"Bachelor's": 0, 'High school': 1, 'PhD': 2}
marital_status_mapping = {'Widowed': 0, 'Divorced': 1, 'Single': 2, 'Married': 3}
loan_purpose_mapping = {'Home': 0, 'Auto': 1, 'Personal': 2, 'Business': 3}
employment_status_mapping = {'Employed': 0, 'Unemployed': 1, 'Self-employed': 2}
payment_history_mapping = {'Excellent': 0, 'Good': 1, 'Fair': 2, 'Poor': 3}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        age = int(request.form['age'])
        gender = request.form['gender']
        education = request.form['education']
        marital_status = request.form['marital_status']
        income = float(request.form['income'])
        credit_score = int(request.form['credit_score'])
        loan_amount = float(request.form['loan_amount'])
        loan_purpose = request.form['loan_purpose']
        employment_status = request.form['employment_status']
        years_at_job = int(request.form['years_at_job'])
        payment_history = request.form['payment_history']
        dti_ratio = float(request.form['dti_ratio'])
        assets_value = float(request.form['assets_value'])
        dependents = int(request.form['dependents'])
        city = request.form['city']
        state = request.form['state']
        country = request.form['country']
        previous_defaults = int(request.form['previous_defaults'])
        marital_status_change = int(request.form['marital_status_change'])

        # Preprocess categorical variables
        gender_encoded = gender_mapping.get(gender, 3)  # Unknown category mapped to 3
        education_encoded = education_mapping.get(education, 3)
        marital_status_encoded = marital_status_mapping.get(marital_status, 4)
        loan_purpose_encoded = loan_purpose_mapping.get(loan_purpose, 4)
        employment_status_encoded = employment_status_mapping.get(employment_status, 3)
        payment_history_encoded = payment_history_mapping.get(payment_history, 4)

        # Handle City, State, Country
        # For simplicity, let's encode them as numerical values using hashing or label encoding
        # Here, we'll use a simple hashing method
        city_encoded = hash(city) % 1000
        state_encoded = hash(state) % 1000
        country_encoded = hash(country) % 1000

        # Create a DataFrame for the input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender_encoded],
            'Education_level': [education_encoded],
            'Marital_status': [marital_status_encoded],
            'Income': [income],
            'Credit_score': [credit_score],
            'Loan_Amount': [loan_amount],
            'Loan_purpose': [loan_purpose_encoded],
            'Employment_Status': [employment_status_encoded],
            'Years_at_current_job': [years_at_job],
            'Payment_history': [payment_history_encoded],
            'Debt_to_Income_Ratio': [dti_ratio],
            'Assets_value': [assets_value],
            'Number_of_dependents': [dependents],
            'City': [city_encoded],
            'State': [state_encoded],
            'Country': [country_encoded],
            'Previous_defaults': [previous_defaults],
            'Marital_Status_change': [marital_status_change]
        })

        # Ensure the order of columns matches the training data
        # Replace the following list with the exact order used during training
        feature_order = [
            'Age', 'Gender', 'Education_level', 'Marital_status', 'Income',
            'Credit_score', 'Loan_Amount', 'Loan_purpose', 'Employment_Status',
            'Years_at_current_job', 'Payment_history', 'Debt_to_Income_Ratio',
            'Assets_value', 'Number_of_dependents', 'City', 'State', 'Country',
            'Previous_defaults', 'Marital_Status_change'
        ]
        input_data = input_data[feature_order]

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Map prediction to risk category
        risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
        risk = risk_mapping.get(prediction, 'Unknown')

        return render_template('result.html', prediction=risk)

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
