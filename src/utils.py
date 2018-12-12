# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing

ONE_HOUR = np.timedelta64(1, 'h')
ONE_DAY = np.timedelta64(1, 'D')
ONE_MONTH = np.timedelta64(1, 'M')
ONE_YEAR = np.timedelta64(1, 'Y')

datatype = [('Timestamp', np.datetime64('1970-01-01T00:00:00')),
            ('FD_Avg', np.float32),
            ('FG_Avg', np.float32),
            ('Patm_Avg', np.float32),
            ('RH_Avg', np.float32),
            ('Text_Avg', np.float32),
            ('WD_MeanUnitVector', np.float32),
            ('WS_Mean', np.float32)]
COLUMNS_WITH_TIMESTAMP = ['Timestamp', 'FD_Avg', 'FG_Avg', 'Patm_Avg', 'RH_Avg', 'Text_Avg', 'WD_MeanUnitVector', 'WS_Mean']
COLUMNS = ['FD_Avg', 'FG_Avg', 'Patm_Avg', 'RH_Avg', 'Text_Avg', 'WD_MeanUnitVector', 'WS_Mean']
NEW_COLUMNS = ['Kb']
DATASETS = ["Moufia", "Possession", "SaintAndre", "SaintLeu", "SaintPierre"]
GRAPHS_PATH = '../graphs/shifted/'


def get_folder_name(dataset_name):
    return '../Data/' + dataset_name + '_2014_2015/'


def get_file_name(dataset_name):
    return dataset_name.lower() + '_2014_2015.csv'


def get_file_path(dataset_name):
    return get_folder_name(dataset_name) + get_file_name(dataset_name)


def split_data_set(timestamps, data, period=ONE_YEAR):
    i = 0
    j = 0
    selected_timestamps = []
    splits = []
    while i < len(timestamps):
        selected_timestamps.append(timestamps[i].round('H'))
        splits.append([])
        while i < len(timestamps) and timestamps[i] < selected_timestamps[j]+period:
            splits[j].append(data[i])
            i += 1
        j += 1

    return selected_timestamps, splits


def average(timestamps, data, period=ONE_HOUR):
    selected_timestamps, split_data = split_data_set(timestamps, data, period)

    averaged_data = []

    for data_one_period in split_data:
        averaged_data.append(sum(data_one_period)/len(data_one_period))

    return selected_timestamps, averaged_data


def average_one_column_parallel(i, data, period):
    columns = COLUMNS + NEW_COLUMNS
    print("\tAveraging " + columns[i])
    if i == 0:
        return average(data['Timestamp'], data[columns[i]], period)
    else:
        return average(data['Timestamp'], data[columns[i]], period)[1]


def average_all_data(data, period=ONE_HOUR):
    columns = COLUMNS + NEW_COLUMNS
    results = Parallel(n_jobs=-1)(delayed(average_one_column_parallel)(i, data, period) for i in range(len(columns)))

    averaged_data = {}
    for i in range(len(columns)):
        if i == 0:
            averaged_data[COLUMNS_WITH_TIMESTAMP[0]] = results[0][0]
            averaged_data[columns[0]] = results[0][1]
        else:
            averaged_data[columns[i]] = results[i]

    return averaged_data


def convert_to_data_frame(data, columns=COLUMNS_WITH_TIMESTAMP):
    d = {}
    for col in columns:
        d[col] = data[col]
    return pd.DataFrame(data=d)


def remove_nocturnal_data(data, start=8, end=16):
    indexes = []
    data = data.reset_index(drop=True)
    for i in range(len(data)):
        timestamp = data['Timestamp'][i]
        if timestamp.hour < start or end < timestamp.hour:
            indexes.append(i)

    data.drop(indexes, inplace=True)
    data = data.reset_index(drop=True)
    return data


def remove_missing_values(data):
    for col in COLUMNS:
        data.drop(data[np.isnan(data[col])].index, inplace=True)
        data.drop(data[data[col] == 0.0].index, inplace=True)

    data = data.reset_index(drop=True)
    return data


def import_dataset(dataset_name, averaged=True):
    file_name = Path("../cachedData/" + dataset_name + ".pkl")
    averaged_file_name = Path("../cachedData/" + dataset_name + "_averaged.pkl")

    if averaged:
        if averaged_file_name.is_file():
            print('Read averaged data from ' + dataset_name + "_averaged.pkl")
            return pd.read_pickle(averaged_file_name)

    if file_name.is_file():
        print('Read data from ' + dataset_name + ".pkl")
        data = pd.read_pickle(file_name)
    else:
        print('Read data from ' + get_file_name(dataset_name))
        data = np.genfromtxt(get_file_path(dataset_name), dtype=datatype, delimiter=',', names=True)
        data = convert_to_data_frame(data)
        data = remove_missing_values(data)
        data['Kb'] = data['FD_Avg']/data['FG_Avg']
        data.to_pickle(file_name)

    if averaged:
        print("Averaging data")
        avg_data = average_all_data(data)
        columns = COLUMNS_WITH_TIMESTAMP + NEW_COLUMNS
        convert_to_data_frame(avg_data, columns).to_pickle(averaged_file_name)
        return avg_data

    return data


def main():
    # generate all cached data
    for dataset in DATASETS[-2:]:
        import_dataset(dataset)


if __name__ == "__main__":
    main()
