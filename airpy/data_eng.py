import pandas as pd
import numpy as np

import joblib

from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def remove_duplicated_rows(dataset):
    print('# Drop duplicated content')
    print('Dataset shape : ', dataset.shape)

    df_results = dataset.drop_duplicates(keep='last', ignore_index=True)

    print('# After process')
    print('Dataset shape : ', df_results.shape)
    return df_results


def missing_values(dataset):
    print('# Drop column and rows containing missing value')
    print('Dataset shape : ', dataset.shape)

    # Remove columns containing too much missing value
    dataset = dataset.dropna(1)
    # Remove rows containing too much missing value
    df_results = dataset.dropna(0)

    print('# After process')
    print('Dataset shape : ', df_results.shape)
    return df_results


def generate_data_dict(dataset, details, path):
    for item in details.items():
        feature_name = item[0]
        feature_type = item[1]['type']
        feature_description = item[1]['description']
        if feature_type == "numerical":
            max_value = dataset[feature_name].max()
            min_value = dataset[feature_name].min()
            details.get(feature_name).update({'max_value': max_value})
            details.get(feature_name).update({'min_value': min_value})
        if feature_type == "categorical":
            unique_value = dataset[feature_name].unique()
            details.get(feature_name).update({'unique_value': list(unique_value)})
    joblib.dump(details, path)


def read_data_dict(path):
    return joblib.load(path)


def adapt_datatype(dataset, path):
    data_dict = read_data_dict(path)
    for item in data_dict.items():
        feature_name = item[0]
        feature_type = item[1]
        if feature_type == "numerical":
            dataset[feature_name] = dataset[feature_name].astype('float64')
        if feature_type == "categorical":
            dataset[feature_name] = dataset[feature_name].astype('category')
    return dataset


def specific_parser(dataset):
    # Mileage : remove 'km'
    dataset['Mileage'] = dataset['Mileage'].str.split('.', expand=True)[0].astype('int32')

    # Online : split into datetime

    pub_date = dataset['Online'].str.split('à', expand=True)[0]
    pub_time = dataset['Online'].str.split('à', expand=True)[1]

    dataset['pub_day'] = pub_date.str.split('/', expand=True)[0].astype('int32')
    dataset['pub_month'] = pub_date.str.split('/', expand=True)[1].astype('int32')
    dataset['pub_year'] = pub_date.str.split('/', expand=True)[2].astype('int32')

    dataset['pub_hour'] = pub_time.str.split('h', expand=True)[0].astype('int32')
    dataset['pub_minute'] = pub_time.str.split('h', expand=True)[1].astype('int32')

    dataset['car_age'] = dataset['pub_year'] - dataset['Model_year']

    dataset['descrip_chevaux'] = dataset['Description'].str.extract('(puissance_fiscale:.*(?=, portes:))')
    dataset['descrip_chevaux'] = dataset['descrip_chevaux'].str.replace('puissance_fiscale: ', '')
    dataset['descrip_chevaux'] = dataset['descrip_chevaux'].astype('int32')

    dataset = dataset.drop('Online', axis=1)

    return dataset


def remove_outlier(df):
    # Price
    print('Shape before removing outlier : ', df.shape)
    target_name = "Price"
    factor = 15
    rows_to_drop = df[np.abs(df[target_name] - df[target_name].mean()) >= (factor * df[target_name].std())].index
    #print(df[np.abs(df[target_name] - df[target_name].mean()) >= (factor * df[target_name].std())][target_name].head())
    df = df.drop(rows_to_drop, axis=0)
    print('Shape after removing outlier : ', df.shape)

    # Mileage
    print('Shape before removing outlier : ', df.shape)
    target_name = "Mileage"
    factor = 5
    rows_to_drop = df[np.abs(df[target_name] - df[target_name].mean()) >= (factor * df[target_name].std())].index
    #print(df[np.abs(df[target_name] - df[target_name].mean()) >= (factor * df[target_name].std())][target_name].head())
    df = df.drop(rows_to_drop, axis=0)
    print('Shape after removing outlier : ', df.shape)

    # car_age
    print('Shape before removing outlier : ', df.shape)
    target_name = "car_age"
    factor = 4
    rows_to_drop = df[np.abs(df[target_name] - df[target_name].mean()) >= (factor * df[target_name].std())].index
    #print(df[np.abs(df[target_name] - df[target_name].mean()) >= (factor * df[target_name].std())][target_name].head())
    df = df.drop(rows_to_drop, axis=0)
    print('Shape after removing outlier : ', df.shape)

    return df


def learn_set(path, target_name):
    dataset = pd.read_csv(path)
    dataset = remove_duplicated_rows(dataset)
    dataset = missing_values(dataset)
    dataset = specific_parser(dataset)
    dataset = remove_outlier(dataset)
    dict_handwritten = {
        "Price": {
            "type": "numerical",
            "description": "Price of the car",
        },
        "Make": {
            "type": "categorical",
            "description": "Brand of the car"
        },
        "Model": {
            "type": "categorical",
            "description": "Model of the car"
        },
        "Model_year": {
            "type": "date",
            "description": "Year of release of the model"
        },
        "Mileage": {
            "type": "numerical",
            "description": "Distance ride by the car"
        },
        "Fuel": {
            "type": "categorical",
            "description": "Type of fuel"
        },
        "Gearbox": {
            "type": "categorical",
            "description": "Type of gearbox"
        },
        "Online": {
            "type": "date",
            "description": "publishing date of the offer"
        },
        "Description": {
            "type": "text",
            "description": "text written by the user about the offer"
        },
        "pub_month": {
            "type": "numerical",
            "description": "Month of publication"
        },
        "pub_year": {
            "type": "numerical",
            "description": "Year of publication"
        },
        "car_age": {
            "type": "numerical",
            "description": "Duration between car model release and offer publication"
        },
        "descrip_chevaux": {
            "type": "numerical",
            "description": "Fiscal Horsepower"
        },
    }
    generate_data_dict(dataset, dict_handwritten, "data_dict.txt")
    dataset = adapt_datatype(dataset, "data_dict.txt")
    print(dataset.columns)
    features = ['Model_year', 'Mileage',
                'pub_month', 'pub_year',
                'car_age', 'descrip_chevaux',
                'Gearbox', 'Fuel',
                'Make', 'Model']
    X = dataset[features]

    # Dummies
    quality_features = ['Gearbox', 'Fuel', 'Make', 'Model']
    quanti_features = ['Model_year', 'Mileage', 'pub_month', 'pub_year', 'car_age', 'descrip_chevaux']
    X_dummy = pd.get_dummies(X[quality_features])
    X = pd.concat([X[quanti_features], X_dummy], axis=1)
    y = dataset[target_name]
    return X, y




