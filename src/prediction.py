# -*- coding: utf-8 -*-

from src.utils import *
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
import datetime
import warnings
warnings.simplefilter("ignore")


COLUMN_TO_PREDICT = 'FD_Avg'
dtype = [('Timestamp', np.datetime64('1970-01-01 00:00:00')), ('FD_Avg', np.float32)]


class BayesSearchCV(BayesSearchCV):
    def _run_search(self, x): raise BaseException('Use newer skopt')


def simplify_scores(scores):
    simplified_scores = {}
    for k, v in scores.items():
        simplified_scores[k] = np.mean(v)
    return simplified_scores


def create_prediction_datasets(data_averaged, t_minus):
    print('Creating prediction data')
    for i in range(1, t_minus):
        for col in COLUMNS:
            new_col_name = col+str(i)
            data_averaged[new_col_name] = data_averaged[col]
            data_averaged[new_col_name] = data_averaged[new_col_name].shift(i)

    prediction = data_averaged[COLUMN_TO_PREDICT].shift(-1)[t_minus:-1]
    attributes = data_averaged[t_minus:-1]

    return prediction, attributes


def random_forest(attributes, prediction):
    x_train, x_test, y_train, y_test = train_test_split(attributes, prediction, random_state=0)

    rf = RandomForestRegressor(n_estimators=200, random_state=42)

    opt = BayesSearchCV(
        rf,
        {
            'n_estimators': Integer(200, 2000),
            'max_features': Categorical(['auto', 'sqrt']),
            'max_depth': Integer(10, 110),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 4),
            'bootstrap': Categorical([True, False]),
        },
        n_iter=32,
        cv=5,
        n_jobs=-1,
        verbose=2
    )
    opt.fit(x_train, y_train)

    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(x_test, y_test))

    params_from_bayes = opt.best_params_

    bayes_rf = RandomForestRegressor(**params_from_bayes)

    scoring = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']

    bayes_scores = cross_validate(bayes_rf, attributes, prediction, scoring=scoring, cv=10)

    print(simplify_scores(bayes_scores))


def evaluate_model(model):
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    # plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    residuals.plot(kind='kde')
    print(residuals.describe())
    # plt.show()


def evaluate_model_prediction(series, exog=None, train_size=48, max_evals=1000, offset=0, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)):
    warnings.simplefilter("ignore")
    observations, predictions = [], []

    start = min(train_size + offset, len(series)-max_evals-train_size)
    end = min(train_size + max_evals + offset, len(series))
    for i in range(start, end):
        train = series[i-train_size:i]
        if exog is not None:
            exog_slice = exog[i-train_size:i]
            print(exog_slice)
            model = SARIMAX(train, exog=exog_slice, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
        else:
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        predictions.append(output[0])
        observations.append(series[i - 1])

    scoring = mean_squared_error(observations, predictions)

    return scoring


def cross_val(series, exog=None, k=10, train_size=48, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)):
    # print("order:", order, "| seasonal order:", seasonal_order, "| train size:", train_size)
    size = len(series)
    chunck_size = int(size/10)

    scores = Parallel(n_jobs=-1)(delayed(evaluate_model_prediction)(series, exog=exog, train_size=train_size, order=order,
                                                                    seasonal_order=seasonal_order, offset=chunck_size*i,
                                                                    max_evals=chunck_size) for i in range(k))
    print("order:", order, "| seasonal order:", seasonal_order, "| mean MSE:", np.average(scores),
          "| MSE std:", np.std(scores))


def try_orders(series, exog=None):
    orders = [
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
        [4, 0, 0]
    ]
    Parallel(n_jobs=1)(delayed(cross_val)(series, train_size=100, order=order) for order in orders)


def main():
    print("Start")

    # prediction, attributes = create_prediction_datasets(avg_data, 5)

    # random_forest(attributes.drop(columns='Timestamp'), prediction)

    print('Training')

    # series = np.genfromtxt('../cachedData/test.csv', dtype=dtype, delimiter=',', names=True)

    # series = avg_data[COLUMN_TO_PREDICT].values.reshape(-1, 1)
    COLUMNS.remove(COLUMN_TO_PREDICT)
    for dataset in DATASETS:
        print(dataset)
        avg_data = import_dataset(dataset, True)
        min_max_scaler = preprocessing.MinMaxScaler()
        series = min_max_scaler.fit_transform(avg_data[COLUMN_TO_PREDICT].values.reshape(-1, 1))
        exog = avg_data[COLUMNS]
        print(sum(series) / len(series))

        cross_val(series)
        print()

    for dataset in DATASETS:
        print(dataset)
        avg_data = import_dataset(dataset, True)
        min_max_scaler = preprocessing.MinMaxScaler()
        series = min_max_scaler.fit_transform(avg_data[COLUMN_TO_PREDICT].values.reshape(-1, 1))
        exog = avg_data[COLUMNS]
        print(sum(series) / len(series))

        series = series[539:]
        cross_val(series)
        print()


if __name__ == "__main__":
    main()
