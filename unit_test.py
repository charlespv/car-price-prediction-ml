import os

import pandas as pd
import autopluspy

def test_1():
    fpath = os.path.join('data', 'Data_cars.csv')
    X, y = autopluspy.data_eng.learn_set(fpath, 'Price')

    assert type(X) == type(pd.DataFrame())
    assert type(y) == type(pd.DataFrame())