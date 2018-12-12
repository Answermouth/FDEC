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
from statsmodels.tsa.stattools import pacf, acf, adfuller
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
import datetime
import warnings
warnings.simplefilter("ignore")


COLUMN_TO_PREDICT = 'Kb'
dtype = [('Timestamp', np.datetime64('1970-01-01 00:00:00')), ('FD_Avg', np.float32)]


def evaluate_model_prediction(series, train_size=48, max_evals=1000, offset=0, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)):
    warnings.simplefilter("ignore")
    observations, predictions = [], []

    start = min(train_size + offset, len(series)-max_evals-train_size)
    end = min(train_size + max_evals + offset, len(series))
    for i in range(start, end):
        train = series[i-train_size:i]

        model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast().iloc[-1]
        # output = model_fit.forecast()[0]

        predictions.append(output)
        observations.append(series.iloc[i - 1])
        # observations.append(series[i - 1])

    scoring = mean_squared_error(observations, predictions)

    plt.plot(observations)
    plt.plot(predictions, color='red')
    plt.show()

    return scoring


def cross_val(series, k=10, train_size=48, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)):
    # print("order:", order, "| seasonal order:", seasonal_order, "| train size:", train_size)
    size = len(series)
    chunck_size = int(size/10)

    scores = Parallel(n_jobs=-1)(delayed(evaluate_model_prediction)(series, train_size=train_size, order=order,
                                                                    seasonal_order=seasonal_order, offset=chunck_size*i,
                                                                    max_evals=chunck_size) for i in range(k))
    print("\torder:", order, "| seasonal order:", seasonal_order, "| RMSE:", np.average(scores),
          "| MSE std:", np.std(scores))
    return scores


def try_orders(series):
    orders = [
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
        [4, 0, 0]
    ]
    outputs = Parallel(n_jobs=1)(delayed(cross_val)(series, train_size=100, order=order) for order in orders)
    return outputs


def get_series(data):
    data = data[539:]
    # data = remove_nocturnal_data(data)
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    series = min_max_scaler.fit_transform(data[COLUMN_TO_PREDICT].values.reshape(-1, 1))"""

    series = data[COLUMN_TO_PREDICT]
    return series


def main():
    print("Start")
    # DATASETS = ["SaintLeu", "SaintPierre"]
    scores = []
    scores2 = []
    for dataset in DATASETS:
        print(dataset)
        data = import_dataset(dataset)
        series = get_series(data)
        # autocorrelation_plot(series)
        # plt.plot(acf(series, nlags=72))
        # plt.plot(pacf(series, nlags=72))

        # plt.figure()

        print("\tLenght:", len(series), "| MIN:", np.min(series), "| MAX:", np.max(series), "| mean:", np.average(series), "| std:", np.std(series))
        score = cross_val(series, order=(1, 2, 0), k=1)
        scores = scores+score
        score = cross_val(series, order=(1, 1, 0), k=1)
        scores2 = scores2 + score
        print()

    print("RMSE:", np.average(scores), "| std:", np.std(scores))
    print("RMSE:", np.average(scores2), "| std:", np.std(scores2))
    #plt.show()


if __name__ == "__main__":
    main()

