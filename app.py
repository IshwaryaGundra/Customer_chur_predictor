# coding: utf-8

import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash
import pickle

app = Flask("__name__")
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Load the training data to get the correct columns
df_1 = pd.read_csv("first_telc.csv")

# Load the trained model and preprocessing objects
model = pickle.load(open("model.sav", "rb"))

# Get the columns used during training
expected_columns = model.feature_names_in_

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/result", methods=['POST'])
def predict():
    '''
    SeniorCitizen
    MonthlyCharges
    TotalCharges
    gender
    Partner
    Dependents
    PhoneService
    MultipleLines
    InternetService
    OnlineSecurity
    OnlineBackup
    DeviceProtection
    TechSupport
    StreamingTV
    StreamingMovies
    Contract
    PaperlessBilling
    PaymentMethod
    tenure
    '''
    
    try:
        # Retrieve form inputs with default values
        def get_input(name):
            value = request.form.get(name, '').strip()
            if value == '':
                return None
            return value

        inputs = [get_input(f'query{i}') for i in range(1, 20)]

        # Check if any input is None
        if any(v is None for v in inputs):
            flash('Please fill all fields.', 'error')
            return redirect(url_for('loadPage'))

        # Convert specific inputs to int or float
        inputs = [
            int(inputs[0]),          # SeniorCitizen
            float(inputs[1]),        # MonthlyCharges
            float(inputs[2]),        # TotalCharges
            inputs[3],               # Gender
            inputs[4],               # Partner
            inputs[5],               # Dependents
            inputs[6],               # PhoneService
            inputs[7],               # MultipleLines
            inputs[8],               # InternetService
            inputs[9],               # OnlineSecurity
            inputs[10],              # OnlineBackup
            inputs[11],              # DeviceProtection
            inputs[12],              # TechSupport
            inputs[13],              # StreamingTV
            inputs[14],              # StreamingMovies
            inputs[15],              # Contract
            inputs[16],              # PaperlessBilling
            inputs[17],              # PaymentMethod
            int(inputs[18])          # Tenure
        ]

        # Create a DataFrame with the input data
        new_df = pd.DataFrame([inputs], columns=[
            'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
            'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'tenure'
        ])

        # Process tenure
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        new_df['tenure_group'] = pd.cut(new_df.tenure, range(1, 80, 12), right=False, labels=labels)
        new_df.drop(columns=['tenure'], axis=1, inplace=True)

        # One-hot encode the DataFrame to match the model's expected format
        new_df_dummies = pd.get_dummies(new_df)

        # Ensure the DataFrame has all expected columns
        for col in expected_columns:
            if col not in new_df_dummies.columns:
                new_df_dummies[col] = 0

        # Reorder columns to match model expectations
        new_df_dummies = new_df_dummies[expected_columns]

        # Make prediction
        prediction = model.predict(new_df_dummies)
        probability = model.predict_proba(new_df_dummies)[:, 1]

        # Interpret results
        if prediction == 1:
            outcome = "This customer is likely to be churned!!"
        else:
            outcome = "This customer is likely to continue!!"

        confidence = f"Confidence: {probability[0] * 100:.2f}%"

        return render_template('result.html', output1=outcome, output2=confidence,
                               **{f'query{i}': request.form[f'query{i}'] for i in range(1, 20)})

    except ValueError as e:
        flash(f"Error: {str(e)}", 'error')
        return redirect(url_for('loadPage'))

if __name__ == "__main__":
    app.run(debug=True)
