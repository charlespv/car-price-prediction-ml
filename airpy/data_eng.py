import pandas as pd

import joblib

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

    dataset = dataset.drop('Online', axis=1)

    return dataset

def learn_set(path, target_name):
    dataset = pd.read_csv(path)
    dataset = remove_duplicated_rows(dataset)
    dataset = missing_values(dataset)
    dataset = specific_parser(dataset)
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
    }
    generate_data_dict(dataset, dict_handwritten, "data_dict.txt")
    dataset = adapt_datatype(dataset, "data_dict.txt")
    print(dataset.columns)
    features = ['Model_year', 'Mileage', 'pub_day', 'pub_month', 'pub_year', 'pub_hour', 'pub_minute', 'car_age']
    X = dataset[features]
    y = dataset[target_name]
    return X, y

