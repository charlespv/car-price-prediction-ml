import joblib
import airpy
import streamlit as st
import numpy as np
import pandas as pd

# Function :
def display_results(results):
    ## Display results
    st.write(f'Predicted price : {results} euros')

# Config
model_filename = 'final_model.sav'
data_dict_path = 'data_dict.txt'

# Call saved file
loaded_model = joblib.load(model_filename)
data_dict = airpy.data_eng.read_data_dict(data_dict_path)
features_name = data_dict.keys()

# Start the app
st.title('Car price estimator')
st.write(' ')
st.write(' ')

# Feature required
st.write('Feature used : ')
st.write(features_name)

# User input
Model_year = st.number_input('Insert the year of the model: ', 1900, value= 2010, step=1)

# Predict dataset
d = {'Model_year': [Model_year]}
x_pred = pd.DataFrame(data=d)

# Summary before prediction
st.write('Summary before prediction ')
st.write(x_pred)

## Running Prediction
if st.button('Run'):
    results = int(loaded_model.predict(x_pred))
    display_results(results)

## Good Deal or not



