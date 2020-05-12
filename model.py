import os
import airpy

# Config
dataset_file_path = os.path.join('data', 'Data_cars.csv')
model_filename = 'final_model.sav'

# Load Data
X, y = airpy.data_eng.learn_set(dataset_file_path, 'Price')

# Split
X_train, y_train, X_test, y_test = airpy.machine_learning.split_train_test(X, y, 0.2)

# Train
quanti_features = ['Model_year', 'Mileage', 'pub_month', 'pub_year', 'car_age', 'descrip_chevaux']
reg = airpy.machine_learning.Model(X_train, y_train, X_test, y_test, quanti_features)

reg.performance()

reg.export(model_filename)



