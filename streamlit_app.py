import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('insuranceML.sav', 'rb'))

st.title('US Health Insurance Prediction')
age = st.slider('Age', 1, 100)
gender = st.number_input('Enter your gender: Male 0, Fmale 1')
bmi = st.number_input('Enter the weight')
children = st.slider('Childrens', 0, 20)
smoker = st.number_input('Are you smoker? no: 0, yes: 1')
options = [1, 2, 3, 4]
region = st.selectbox('Choose your region: southeast 1, southwest 2, northwest 3, northeast 4', options)

input_data = [age, gender, bmi, children, smoker, region]

result = ''
if st.button('Result'):
  input_data = np.asarray(input_data).reshape(1,-1)
  prediction = model.predict(input_data)
  result = prediction[0]
  

st.success(result)
