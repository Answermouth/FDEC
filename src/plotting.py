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
    for col in COLUMNS:
        # draw_box_plot_yearly(timestamps, data[col], dataset_prefix + col)
        draw_box_plot_monthly(timestamps, data[col], dataset_prefix + col)


def calculate_correlations(data):
    correlations = np.zeros((len(COLUMNS), len(COLUMNS)))
    for i in range(len(COLUMNS)-1):
        col1 = COLUMNS[i]
        for j in range(i+1, len(COLUMNS)):
            col2 = COLUMNS[j]
            correlations[i][j] = np.corrcoef(data[col1], data[col2])[0][1]
    return correlations


def plot_correlations(data, dataset_prefix=''):
    correlations = calculate_correlations(data)
    plt.clf()
    sns.heatmap(correlations, cmap=CMAP, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=COLUMNS, yticklabels=COLUMNS, annot=True)
    plt.savefig(GRAPHS_PATH + dataset_prefix + 'correlations.png')
    print('\tSaved plot: ' + dataset_prefix + 'correlations.png')


def main():
    for dataset_name in DATASETS:
        print('Working on dataset: ' + dataset_name)
        data = import_dataset(dataset_name, False)
        draw_all_box_plots(data, dataset_name+'_')
        plot_correlations(data, dataset_name + '_')
        print('Averaging data')
        plot_correlations(average_all_data(data), dataset_name + '_averaged_')
        print('')


if __name__ == "__main__":
    main()
