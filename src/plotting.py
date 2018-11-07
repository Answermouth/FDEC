# -*- coding: utf-8 -*-

import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ONE_HOUR = numpy.timedelta64(1,'h')
ONE_YEAR = numpy.timedelta64(365,'D')

figureCount = 0

datatype = [('Timestamp',numpy.datetime64('1970-01-01T00:00:00')),
            ('FD_Avg',numpy.float32),
            ('FG_Avg',numpy.float32),
            ('Patm_Avg',numpy.float32),
            ('RH_Avg',numpy.float32),
            ('Text_Avg',numpy.float32),
            ('WD_MeanUnitVector',numpy.float32),
            ('WS_Mean',numpy.float32)]
foldername='../Data/Moufia_2014_2015/'
filename ='moufia_2014_2015.csv'
filepath = foldername + filename

def average(timeline, values, scale = ONE_HOUR):
    averagedValues = []
    simplifiedTimeline = []
    i = 0
    while i<len(timeline):
        summedValues = 0.0
        valueCount = 0.0
        hour = timeline[i]
        while i<len(timeline) and timeline[i]<hour+scale:
            summedValues += values[i]
            valueCount += 1
            i += 1
        averagedValues.append(summedValues/valueCount)
        simplifiedTimeline.append(hour)
    
    return simplifiedTimeline, averagedValues

def drawBoxPlot(data, title=''):
    global figureCount
    x = []
    x.append(data[:522592])
    x.append(data[522592:])
    
    figureCount+=1
    plt.figure(figureCount)
    
    plt.boxplot(x, sym='')
    plt.title(title)
    plt.draw()

def main():    
    data = numpy.genfromtxt(filepath, dtype=datatype, delimiter=',', names=True)
    #houraveraged = numpy.mean(data['FD_Avg'][:1022400].reshape(-1,1440), axis=1)
    #plt.plot(houraveraged[:365])
    
    #x, y = average(data['Timestamp'], data['Patm_Avg'], ONE_YEAR)
    
    drawBoxPlot(data['FD_Avg'],'FD_Avg')
    drawBoxPlot(data['FG_Avg'],'FG_Avg')
    drawBoxPlot(data['Patm_Avg'],'Patm_Avg')
    
    #plt.plot(x, y,'r,')    
    #plt.plot(data['Timestamp'],data['FG_Avg'],'b,')
    #plt.grid(True)
    #plt.title("Direct illumination")
    #plt.xlabel("Timestamp")
    #plt.ylabel("FD_Avg")
    #plt.savefig('FD_Avg.png')
    
    #plt.draw()
    

if __name__ == "__main__":
    main()