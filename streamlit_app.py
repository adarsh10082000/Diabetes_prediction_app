import streamlit as st
import numpy as np
import pickle

# Load the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def diabetes_prediction(input_data):
    try:
        # Convert to numpy array
        input_data_as_numpy_array = np.asarray(input_data, dtype=float)

        # Reshape for single prediction
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Make prediction
        prediction = loaded_model.predict(input_data_reshaped)

        if prediction[0] == 0:
            return 'The person is **not diabetic**'
        else:
            return 'The person is **diabetic**'

    except Exception as e:
        return f"Error in prediction: {str(e)}"

def main():
    st.title('Diabetes Prediction Web App')
    
    # User input
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        input_values = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        try:
            # Convert input values to float
            input_values = list(map(float, input_values))
            diagnosis = diabetes_prediction(input_values)
        except ValueError:
            diagnosis = 'Please enter valid numerical inputs for all fields.'
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
