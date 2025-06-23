import streamlit as st
import numpy as np
import pickle

# Load the model once and cache it
@st.cache_resource
def load_model():
    with open('./trained_model.sav', 'rb') as f:
        return pickle.load(f)
    # Load the model globally
    loaded_model = load_model()

def diabetes_prediction(input_data):
    try:
        # Convert input to numpy array
        input_data_as_numpy_array = np.asarray(input_data, dtype=float)

        # Reshape for prediction
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Predict
        prediction = loaded_model.predict(input_data_reshaped)

        if prediction[0] == 0:
            return 'The person is **not diabetic**.'
        else:
            return 'The person is **diabetic**.'

    except Exception as e:
        return f"Error in prediction: {str(e)}"

def main():
    st.title('🩺 Diabetes Prediction Web App')

    # Inputs
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the Person')

    diagnosis = ''

    if st.button('🔍 Diabetes Test Result'):
        input_values = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                        Insulin, BMI, DiabetesPedigreeFunction, Age]

        if all(input_values):  # Check if none are empty
            try:
                input_values = list(map(float, input_values))
                diagnosis = diabetes_prediction(input_values)
            except ValueError:
                diagnosis = '❌ Please enter valid numerical values.'
        else:
            diagnosis = '⚠️ All fields are required.'

    if diagnosis:
        st.success(diagnosis)

if __name__ == '__main__':
    main()
