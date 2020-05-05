from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

from sklearn.metrics import mean_squared_error, r2_score

import joblib

def split_train_test(X, y, ratio_size):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=ratio_size, random_state=42)
    return x_train, y_train, x_test, y_test


class Model:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.mdl = LinearRegression().fit(x_train, y_train)
        #self.mdl = DummyRegressor(strategy="mean").fit(x_train, y_train)
        self.y_pred = self.mdl.predict(x_test)

    def performance(self):
        # The mean squared error
        print('Mean squared error: ', mean_squared_error(self.y_test, self.y_pred))
        # The coefficient of determination: 1 is perfect prediction
        print('Coefficient of determination: ', r2_score(self.y_test, self.y_pred))

    def predict(self, x_pred):
        y_pred = self.mdl.predict(x_pred)
        return y_pred

    def export(self, path):
        # save the model to disk
        joblib.dump(self.mdl, path)


