import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer, quantile_transform, PowerTransformer

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import joblib


def split_train_test(X, y, ratio_size):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=ratio_size, random_state=42)
    return x_train, y_train, x_test, y_test


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class Model:
    def __init__(self, x_train, y_train, x_test, y_test, quanti_features):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        #Scaled
        self.x_train_scaled = x_train
        self.x_test_scaled = x_test
        self.scale_mdl = StandardScaler().fit(x_train[quanti_features])
        self.x_train_scaled[quanti_features] = self.scale_mdl.transform(self.x_train[quanti_features])
        self.x_test_scaled[quanti_features] = self.scale_mdl.transform(self.x_test[quanti_features])
        # self.mdl = LinearRegression().fit(self.x_train_scaled, self.y_train)
        self.mdl = Ridge().fit(self.x_train_scaled, self.y_train)
        # self.mdl = DummyRegressor(strategy="mean").fit(self.x_train, self.y_train)
        self.y_pred = self.mdl.predict(self.x_test_scaled)
        """
        self.regr_trans = TransformedTargetRegressor(regressor=Ridge(),
                                                     transformer=QuantileTransformer(n_quantiles=1000, 
                                                     output_distribution='normal')).fit(self.x_train_scaled, self.y_train)
        """
        self.regr_trans = TransformedTargetRegressor(regressor=Ridge(),
                                                     transformer=PowerTransformer(method='box-cox')).fit(self.x_train_scaled, self.y_train)
        self.y_pred_trans = self.regr_trans.predict(self.x_test_scaled)

    def performance(self):
        # The mean squared error
        print('Mean squared error: ', mean_squared_error(self.y_test, self.y_pred))
        # The coefficient of determination: 1 is perfect prediction
        print('Coefficient of determination: ', r2_score(self.y_test, self.y_pred))
        print('MAPE: ', mean_absolute_percentage_error(self.y_test, self.y_pred))
        print('MAPE Trans: ', mean_absolute_percentage_error(self.y_test, self.y_pred_trans))
        print('MAE Trans: ', mean_absolute_error(self.y_test, self.y_pred_trans))
        print('MSE Trans : ', mean_squared_error(self.y_test, self.y_pred_trans))

    def predict(self, x_pred):
        y_pred = self.mdl.predict(x_pred)
        return y_pred

    def export(self, path):
        # save the model to disk
        joblib.dump(self.mdl, path)
