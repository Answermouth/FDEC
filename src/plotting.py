# -*- coding: utf-8 -*-

import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ONE_HOUR = numpy.timedelta64(1,'h')
ONE_YEAR = numpy.timedelta64(365,'D')
GRAPHS_PATH = '../graphs/'

figureCount = 0

datatype = [('Timestamp',numpy.datetime64('1970-01-01T00:00:00')),
            ('FD_Avg',numpy.float32),
            ('FG_Avg',numpy.float32),
            ('Patm_Avg',numpy.float32),
            ('RH_Avg',numpy.float32),
            ('Text_Avg',numpy.float32),
            ('WD_MeanUnitVector',numpy.float32),
            ('WS_Mean',numpy.float32)]
foldername = '../Data/Moufia_2014_2015/'
filename = 'moufia_2014_2015.csv'
filepath = foldername + filename


def average(timeline, values, scale=ONE_HOUR):
    averaged_values = []
    simplified_timeline = []
    i = 0
    while i < len(timeline):
        summed_values = 0.0
        value_count = 0.0
        hour = timeline[i]
        while i < len(timeline) and timeline[i] < hour+scale:
            summed_values += values[i]
            value_count += 1
            i += 1
        averaged_values.append(summed_values/value_count)
        simplified_timeline.append(hour)
    
    return simplified_timeline, averaged_values


def draw_box_plot(data, title=''):
    global figureCount
    x = [data[:522592], data[522592:]]
    
    figureCount += 1
    plt.figure(figureCount)
    
    plt.boxplot(x, sym='')
    plt.title(title)
    plt.draw()
    plt.savefig(GRAPHS_PATH + title + '.png')


def main():    
    data = numpy.genfromtxt(filepath, dtype=datatype, delimiter=',', names=True)

    draw_box_plot(data['FD_Avg'],'FD_Avg')
    draw_box_plot(data['FG_Avg'],'FG_Avg')
    draw_box_plot(data['Patm_Avg'],'Patm_Avg')


if __name__ == "__main__":
    main()
