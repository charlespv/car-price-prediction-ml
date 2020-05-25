import numpy as np

from sklearn.preprocessing import QuantileTransformer, quantile_transform, PowerTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, ElasticNet, LassoCV, LassoLars
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.compose import TransformedTargetRegressor

from sklearn.metrics import r2_score, make_scorer

from sklearn.inspection import permutation_importance

import joblib
import matplotlib.pyplot as plt


def split_train_test(X, y, ratio_size):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=ratio_size, random_state=42)
    return x_train, y_train, x_test, y_test


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class Model:
    def __init__(self, x_train, y_train, x_test, y_test, quanti_features):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_train_scaled = x_train
        self.x_test_scaled = x_test
        self.scale_mdl = StandardScaler().fit(x_train[quanti_features])
        self.x_train_scaled[quanti_features] = self.scale_mdl.transform(self.x_train[quanti_features])
        self.x_test_scaled[quanti_features] = self.scale_mdl.transform(self.x_test[quanti_features])
        self.mdl = Ridge().fit(self.x_train_scaled, self.y_train)
        self.y_pred = self.mdl.predict(self.x_test_scaled)
        self.regr_trans = TransformedTargetRegressor(regressor=RandomForestRegressor(),
                                                     transformer=PowerTransformer(method='box-cox')).fit(self.x_train_scaled, self.y_train)
        self.y_pred_trans = self.regr_trans.predict(self.x_test_scaled)

    def grid_search(self):
        # Instantiate models
        linear_mdl = LinearRegression()
        ridge_mdl = Ridge()
        sgd_mdl = SGDRegressor()
        elastic_mdl = ElasticNet()
        lassocv_mdl = LassoCV()
        lassolars_mdl = LassoLars()
        rf_mdl = RandomForestRegressor()
        gradient_mdl = GradientBoostingRegressor()

        # Grid Config
        pipe = Pipeline([('regressor', linear_mdl)])
        search_space = [{'regressor': [linear_mdl]},
                        {'regressor': [ridge_mdl]},
                        {'regressor': [sgd_mdl]},
                        {'regressor': [rf_mdl]},]

        """
        {'regressor': [elastic_mdl]},
        {'regressor': [lassocv_mdl]},
        {'regressor': [lassolars_mdl]},
        {'regressor': [gradient_mdl]}
        """
        mae_scorer = make_scorer(mean_absolute_percentage_error, False)
        reg = GridSearchCV(pipe, search_space, cv=3, verbose=2, scoring=mae_scorer)
        benchmark = reg.fit(self.x_train_scaled, self.y_train)
        print(benchmark.best_estimator_.get_params()['regressor'])


    def cross_val(self, nb_fold):
        mae_scorer = make_scorer(mean_absolute_percentage_error)
        scores = cross_val_score(self.regr_trans, self.x_train_scaled,
                                 self.y_train, cv=nb_fold, scoring=mae_scorer)
        print(scores)

        return scores

    def performance(self):
        # The coefficient of determination: 1 is perfect prediction
        print('Coefficient of determination: ', r2_score(self.y_test, self.y_pred))
        print('MAPE Ridge : ', mean_absolute_percentage_error(self.y_test, self.y_pred))
        print('MAPE Power Transformed + RF : ', mean_absolute_percentage_error(self.y_test, self.y_pred_trans))

    def permutation_importance(self):
        result = permutation_importance(self.regr_trans, self.x_train_scaled, self.y_train, n_repeats=3,
                                        random_state=42, n_jobs=2)
        sorted_idx = result.importances_mean.argsort()[0:20]

        fig, ax = plt.subplots()
        ax.boxplot(result.importances[sorted_idx].T,
                   vert=False, labels=self.x_train_scaled.columns[sorted_idx])
        ax.set_title("Permutation Importances (test set)")
        fig.tight_layout()
        plt.show()

    def predict(self, x_pred):
        y_pred = self.mdl.predict(x_pred)
        return y_pred

    def export(self, reg_path, std_path, features_path):
        # save the model to disk
        joblib.dump(self.mdl, reg_path)

        # save the model to disk
        joblib.dump(self.scale_mdl, std_path)

        # save features column needed
        joblib.dump(list(self.x_train.columns), features_path)
