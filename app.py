import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer

# Load the pickled model, encoders, and scaler
with open('model/cat_encoder.pickle', 'rb') as f:
    categorical_encoder = pickle.load(f)
with open('model/label_encoder.pickle', 'rb') as f:
    label_encoder = pickle.load(f)
with open('model/scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)
with open('model/xgboost_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Function for preprocessing
def preprocess_input(data):
    categorical_features = ['gender', 'education', 'marital_status', 'loan_purpose', 'employment_status', 'payment_history', 'city', 'state', 'country']
    transformer = ColumnTransformer([("ordinal", categorical_encoder, categorical_features)], remainder="passthrough")
    transformed_data = transformer.fit_transform(data)
    scaled_data = scaler.transform(transformed_data)
    return scaled_data

st.set_page_config(
    layout="wide",
)



# Title of the app
st.title('Financial Risk Prediction')

# Use columns to organize personal information
st.header("Personal Information")
col1, col2, col3 = st.columns([1, 1, 1])  # Set equal width for columns

with col1:
    age = st.slider('Age', min_value=18, max_value=69, value=23)
    gender = st.selectbox('Gender', ['Male', 'Female', 'Non-binary'])

with col2:
    education = st.selectbox('Education Level', ["Bachelor's", "High school", "PhD", "Master's"])
    marital_status = st.selectbox('Marital Status', ['Widowed', 'Divorced', 'Single', 'Married'])

with col3:
    dependents = st.number_input('Number of Dependents', min_value=0, max_value=4, step=1)
    marital_status_change = st.number_input('Marital Status Change', min_value=0, max_value=2, step=1)

# Financial Information section
st.header("Financial Information")
col4, col5, col6 = st.columns([1, 1, 1])  # Set equal width for columns

with col4:
    income = st.number_input('Income (in USD)', min_value=20000, max_value=120000, step=1000)
    credit_score = st.number_input('Credit Score', min_value=600, max_value=799, step=1)

with col5:
    loan_amount = st.number_input('Loan Amount (in USD)', min_value=5000, max_value=50000, step=500)
    loan_purpose = st.selectbox('Loan Purpose', ['Home', 'Auto', 'Personal', 'Business'])

with col6:
    employment_status = st.selectbox('Employment Status', ['Employed', 'Unemployed', 'Self-employed'])
    years_at_job = st.slider('Years at Current Job', min_value=0, max_value=19, value=0)

# Further financial details
col7, col8, col9 = st.columns([1, 1, 1])

with col7:
    payment_history = st.selectbox('Payment History', ['Excellent', 'Good', 'Fair', 'Poor'])

with col8:
    dti_ratio = st.slider('Debt-to-Income Ratio', min_value=0.1, max_value=0.6, step=0.01, value=0.1)

with col9:
    assets_value = st.number_input('Assets Value (in USD)', min_value=20100, max_value=300000, step=1000)

# Location Information section
st.header("Location Information")
col10, col11, col12 = st.columns([1, 1, 1])

with col10:
    city = st.text_input('City')

with col11:
    state = st.text_input('State')

with col12:
    country = st.text_input('Country')

# Previous defaults
previous_defaults = st.number_input('Previous Defaults', min_value=0, max_value=4, step=1)


# Predict button
if st.button("Predict Financial Risk"):
    # Create a dataframe with the input data
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'education': [education],
        'marital_status': [marital_status],
        'dependents': [dependents],
        'marital_status_change': [marital_status_change],
        'income': [income],
        'credit_score': [credit_score],
        'loan_amount': [loan_amount],
        'loan_purpose': [loan_purpose],
        'employment_status': [employment_status],
        'years_at_job': [years_at_job],
        'payment_history': [payment_history],
        'dti_ratio': [dti_ratio],
        'assets_value': [assets_value],
        'city': [city],
        'state': [state],
        'country': [country],
        'previous_defaults': [previous_defaults]
    })

    # Preprocess the input data
    processed_data = preprocess_input(input_data)

    # Make the prediction
    prediction = model.predict(processed_data)

    # Decode the label if necessary
    prediction_label = label_encoder.inverse_transform(prediction)[0]

    # Display the prediction
    st.success(f"Predicted Loan Risk: {prediction_label}")
