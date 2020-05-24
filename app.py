import pprint
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
import pandas as pd
import joblib

import autopluspy

import streamlit as st



# Function :
def display_results(results):
    ## Display results
    st.title(f'Predicted price : {results} euros')

# Config
reg_mdl_filename = 'final_reg.sav'
std_mdl_filename = 'final_std.sav'
features_filename = 'final_features.sav'

data_dict_path = 'data_dict.txt'

# Call saved file
loaded_model = joblib.load(reg_mdl_filename)
loaded_std = joblib.load(std_mdl_filename)
features_list = joblib.load(features_filename)
data_dict = autopluspy.data_eng.read_data_dict(data_dict_path)
features_name = list(data_dict.keys())
features_name = features_name[1:]

# Start the app
st.title('Car price estimator')
st.write(' ')
st.write(' ')

d = {}
temp = {}
quanti_features = []
quality_features = []


for feature in features_list:
    d[feature] = 0

for feature in features_name:
    if data_dict[feature]['type'] == 'numerical':
        key = feature
        value = st.number_input(f'Insert {feature}:')
        temp[key] = value
        quanti_features.append(feature)
    if data_dict[feature]['type'] == 'categorical':
        key = feature
        value = st.selectbox(f'Select {feature}: ', tuple(data_dict[feature]['unique_value']))
        temp[key] = value
        quality_features.append(feature)


## Running Prediction
if st.button('Run'):
    x_temp = pd.DataFrame(data=temp, index=[0])
    x_temp_dummy = pd.get_dummies(x_temp[quality_features])

    x_pred = pd.DataFrame(data=d, index=[0], columns=features_list)
    x_pred = x_pred.drop(list(x_temp_dummy.columns), axis=1)
    x_pred = pd.concat([x_pred, x_temp_dummy], axis=1)

    x_pred[quanti_features] = loaded_std.transform(x_temp[quanti_features])

    #x_pred_updated = pd.concat([x_pred[quanti_features], x_pred_dummy], axis=1)

    # Summary before prediction
    st.write('Summary before prediction ')
    st.write(x_pred)

    results = int(loaded_model.predict(x_pred))
    display_results(results)

## Good Deal or not



