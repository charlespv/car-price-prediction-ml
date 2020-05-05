import pandas as pd
from sklearn.model_selection import train_test_split

import joblib


def generate_data_dict(dataset, path):
    columns = dataset.columns
    joblib.dump(columns, path)


def read_data_dict(path):
    return joblib.load(path)


def learn_set(path, target_name):
    df = pd.read_csv(path)
    features = ['Model_year']
    X = df[features]
    y = df[target_name]
    return X, y


def split_train_test(X, y, ratio_size):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=ratio_size, random_state=42)
    return x_train, y_train, x_test, y_test

