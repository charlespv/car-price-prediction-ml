import pandas as pd

def get_shape(df):
    return df.shape

def learn_set(fpath, target_name):
    df = pd.read_csv(fpath)
    features = ['Make', 'Model', 'Model_year', 'Mileage', 'Fuel', 'Gearbox']
    X = df[features]
    y = df[target_name]
    return X, y