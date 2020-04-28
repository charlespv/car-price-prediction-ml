import os

import pandas as pd
import airpy

# Load Data
fpath = os.path.join('data', 'Data_cars.csv')
X, y = airpy.data_eng.learn_set(fpath, 'Price')

print(X.shape, y.shape)

# Split

# Train

# Export model