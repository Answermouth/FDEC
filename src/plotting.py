# -*- coding: utf-8 -*-

import matplotlib
import seaborn as sns
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils import *

CMAP = sns.diverging_palette(220, 10, as_cmap=True)


def draw_box_plot_yearly(timestamps, data, title=''):
    t, x = split_data_set(timestamps, data)
    plt.clf()
    plt.boxplot(x, sym='')
    plt.title(title)
    plt.draw()
    plt.savefig(GRAPHS_PATH + title + '.png')
    print('\tSaved plot: ' + title + '.png')


def draw_box_plot_monthly(timestamps, data, title=''):
    t, x = split_data_set(timestamps, data, period=ONE_MONTH)
    plt.clf()
    plt.boxplot(x, sym='')
    plt.title(title)
    plt.draw()
    plt.savefig(GRAPHS_PATH + title + '_months.png')
    print('\tSaved plot: ' + title + '_months.png')


def draw_all_box_plots(data, dataset_prefix=''):
    timestamps = data['Timestamp']
    columns = COLUMNS + NEW_COLUMNS
    for col in columns:
        draw_box_plot_yearly(timestamps, data[col], dataset_prefix + col)
        draw_box_plot_monthly(timestamps, data[col], dataset_prefix + col)


def calculate_correlations(data):
    columns = COLUMNS + NEW_COLUMNS
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
    columns = COLUMNS + NEW_COLUMNS
    sns.heatmap(correlations, cmap=CMAP, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=columns, yticklabels=columns, annot=True)
    plt.savefig(GRAPHS_PATH + dataset_prefix + 'correlations.png')
    print('\tSaved plot: ' + dataset_prefix + 'correlations.png')


def main():
    for dataset_name in DATASETS:
        print('Working on dataset: ' + dataset_name)
        data = import_dataset(dataset_name, False)
        draw_all_box_plots(data, dataset_name+'_')
        plot_correlations(data, dataset_name + '_')
        print('Averaging data')
        avg_data = import_dataset(dataset_name)
        avg_data = remove_missing_values(avg_data)
        plot_correlations(avg_data, dataset_name + '_averaged_')
        print('Removing nocturnal data')
        day_data = remove_nocturnal_data(avg_data)
        draw_all_box_plots(day_data, dataset_name + '_day_')
        plot_correlations(day_data, dataset_name + '_day_')
        print('')


if __name__ == "__main__":
    main()
