import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


## Loading the Trained model
model = tf.keras.models.load_model('Model/model.h5')

## Loading the Encoders
with open('Model/label_encoder.pkl', 'rb') as file :
    label_encoder_gender = pickle.load(file)

with open('Model/onehot_encoder.pkl', 'rb') as file :
    onehot_encoder_geography = pickle.load(file)

## Loading Scaler 
with open('Model/scaler.pkl', 'rb') as file :
    scaler = pickle.load(file)


## Streamlit App
st.title('Customer Churn Prediction')

## User Input
geography = st.selectbox('Geography', onehot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


## Preparing Input Data
input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

## Encoding Geography
geography_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns = onehot_encoder_geography.get_feature_names_out(['Geography']))

## Clubbing encoded geography data with input data
input_data = pd.concat([input_data.reset_index(drop=True), geography_encoded_df], axis=1)

## Scaling the data 
scaled_input_data = scaler.transform(input_data)

## Churn Prediction
prediction = model.predict(scaled_input_data)
prediction_probability = prediction[0][0]

st.write(f'Churn Probability : {prediction_probability:.2f}')

if prediction_probability > 0.5 : 
    st.write('The Customer is likely to Churn.')
else :
    st.write('The Customer is not likely to Churn.')