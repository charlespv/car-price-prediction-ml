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
reg = airpy.machine_learning.Model(X_train, y_train, X_test, y_test)

reg.performance()

reg.export(model_filename)



