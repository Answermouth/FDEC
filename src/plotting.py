# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import seaborn as sns
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ONE_HOUR = np.timedelta64(1,'h')
ONE_YEAR = np.timedelta64(365,'D')
GRAPHS_PATH = '../graphs/'

DATASETS = ["Moufia", "Possession", "SaintAndre", "SaintLeu", "SaintPierre"]
CMAP = sns.diverging_palette(220, 10, as_cmap=True)

datatype = [('Timestamp', np.datetime64('1970-01-01T00:00:00')),
            ('FD_Avg', np.float32),
            ('FG_Avg', np.float32),
            ('Patm_Avg', np.float32),
            ('RH_Avg', np.float32),
            ('Text_Avg', np.float32),
            ('WD_MeanUnitVector', np.float32),
            ('WS_Mean', np.float32)]
columns = ['FD_Avg', 'FG_Avg', 'Patm_Avg', 'RH_Avg', 'Text_Avg', 'WD_MeanUnitVector', 'WS_Mean']

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
        selected_timestamps.append(timestamps[i])
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


def average_all_data(data, period=ONE_HOUR):
    timestamps = data['Timestamp']
    averaged_data = {}
    for col in columns:
        averaged_data['Timestamp'], averaged_data[col] = average(timestamps, data[col], period)

    return averaged_data


def draw_box_plot(timestamps, data, title=''):
    t, x = split_data_set(timestamps, data)
    plt.clf()
    plt.boxplot(x, sym='')
    plt.title(title)
    plt.draw()
    plt.savefig(GRAPHS_PATH + title + '.png')
    print('\tSaved plot: ' + title + '.png')


def draw_all_box_plots(data, dataset_prefix=''):
    timestamps = data['Timestamp']
    for col in columns:
        draw_box_plot(timestamps, data[col], dataset_prefix + col)


def calculate_correlations(data):
    correlations = np.zeros((len(columns), len(columns)))
    for i in range(len(columns)-1):
        col1 = columns[i]
        for j in range(i+1, len(columns)):
            col2 = columns[j]
            correlations[i][j] = np.corrcoef(data[col1], data[col2])[0][1]
    return correlations


def plot_correlations(data, dataset_prefix=''):
    correlations = calculate_correlations(data)
    plt.clf()
    sns.heatmap(correlations, cmap=CMAP, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=columns, yticklabels=columns, annot=True)
    plt.savefig(GRAPHS_PATH + dataset_prefix + 'correlations.png')
    print('\tSaved plot: ' + dataset_prefix + 'correlations.png')


def main():
    for dataset_name in DATASETS:
        print('Working on dataset: ' + dataset_name)
        data = np.genfromtxt(get_file_path(dataset_name), dtype=datatype, delimiter=',', names=True)
        draw_all_box_plots(data, dataset_name+'_')
        plot_correlations(data, dataset_name + '_')
        print('Averaging data')
        plot_correlations(average_all_data(data), dataset_name + '_averaged_')
        print('')


if __name__ == "__main__":
    main()
