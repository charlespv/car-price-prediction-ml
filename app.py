import pandas as pd
import airpy

# Example code
df = pd.read_csv('data/Data_cars.csv')
print(airpy.data_eng.get_shape(df))