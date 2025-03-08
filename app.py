import streamlit as st
import joblib
import pandas as pd

# Load the model
model_path = 'artifacts/model_trainer/model.joblib'
random_forest = joblib.load(model_path)

# Define the categories and custom encoder
categories = ['No Failure', 'Heat Dissipation Failure', 'Power Failure', 
              'Overstrain Failure', 'Tool Wear Failure', 'Random Failures']
custom_encoder = {i: cat for i, cat in enumerate(categories)}

# Title of the web app
st.title("Machine Predictive Maintenance Classification")

# Input fields for user data
col1, col2 = st.columns(2)

with col1:
    selected_type = st.selectbox('Select a Type', ['Low', 'Medium', 'High'])
    type_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    selected_type = type_mapping[selected_type]

with col2:
    air_temperature = st.number_input('Air temperature [K]', min_value=0.0, format="%.2f")

with col1:
    process_temperature = st.number_input('Process temperature [K]', min_value=0.0, format="%.2f")

with col2:
    rotational_speed = st.number_input('Rotational speed [rpm]', min_value=0.0, format="%.2f")

with col1:
    torque = st.number_input('Torque [Nm]', min_value=0.0, format="%.2f")

with col2:
    tool_wear = st.number_input('Tool wear [min]', min_value=0.0, format="%.2f")

# Prediction logic
if st.button('Predict Failure'):
    try:
        # Create a DataFrame for the prediction input
        input_data = pd.DataFrame([[selected_type, air_temperature, 
                                    process_temperature, rotational_speed,
                                    torque, tool_wear]], 
                                  columns=['Type', 'Air temperature [K]', 'Process temperature [K]', 
                                           'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])

        # Make a prediction
        prediction = random_forest.predict(input_data)

        # Interpret prediction
        failure_pred = custom_encoder.get(prediction[0], "Unknown Failure Type")

        st.success(f"Predicted Failure Type: {failure_pred}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
