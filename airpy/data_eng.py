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
    dataset = dataset.dropna(1, thresh=dataset.shape[0]*0.80)
    # Remove rows containing too much missing value
    df_results = dataset.dropna(0, thresh=dataset.shape[0] * 0.80)

    print('# After process')
    print('Dataset shape : ', df_results.shape)
    return df_results

def generate_data_dict(dataset, path):
    columns = dataset.columns
    joblib.dump(columns, path)


def read_data_dict(path):
    return joblib.load(path)


def learn_set(path, target_name):
    dataset = pd.read_csv(path)
    dataset = remove_duplicated_rows(dataset)
    dataset = missing_values(dataset)
    features = ['Model_year']
    X = dataset[features]
    y = dataset[target_name]
    return X, y

