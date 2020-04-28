import os

import pandas as pd
import airpy

def test_1():
    fpath = os.path.join('data', 'Data_cars.csv')
    X, y = airpy.data_eng.learn_set(fpath, 'Price')

    assert type(X) == type(pd.DataFrame())
    assert type(y) == type(pd.DataFrame())