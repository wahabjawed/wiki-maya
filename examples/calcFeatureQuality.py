import glob
import multiprocessing
import os
from itertools import islice

import language_check
import numpy
import pandas as pd
from matplotlib import pyplot
from textstat import textstat
# import textstat
from examples.TrustScore import TrustScore
from maya.nltk import util
from readcalc import readcalc
from mwapi import Session
from maya.extractors import api


def evaluateQuality(x):
    if x == 'Stub':
        return arr[0]
    if x == 'Start':
        return arr[1]
    if x == 'C':
        return arr[2]
    if x == 'B':
        return arr[3]
    if x == 'GA':
        return arr[4]
    if x == 'FA':
        return arr[5]

def plotGraph(x,y):

    plot = pyplot.plot(x, y, 'b-')

    # for row in series.iterrows():
    #     if row[1][graph_for] > 1000:
    #         pyplot.annotate(row[1]['pageid'], (row[1]['timestamp'], row[1][graph_for]))

    pyplot.xticks(rotation=45, ha='right')
    pyplot.xlabel("Timestamp")
    pyplot.ylabel("Trust Score")
    ax = pyplot.gca()
    ax = pyplot.gca()

    pyplot.yticks(numpy.arange(0,0.3, 0.1))
    #ax.yaxis.set_major_locator(pyplot.MaxNLocator(15))
    ax.set_xticklabels([])
    pyplot.show()

if __name__ == "__main__":
    rev_xl = pd.read_csv("readscore/data.csv",
                         dtype={0: 'int32', 1: 'int32', 2: 'object'})
    id = 3188459.0
    arr = [round(x,2) for x in numpy.linspace(start=0.1, stop=1.0, num=6)]
    print(arr)
    #rev_xl['quality_score'] = rev_xl['rating'].apply(evaluateQuality)
    #rev_xl['contribQuality'] = round(rev_xl['userContrib'] * rev_xl['quality_score'], 2)
    #rev_xl.to_csv("readscore/data.csv")

    rev_xl = rev_xl[rev_xl['userID'] == id]
    #rr = rev_xl.groupby('pageid')['pageid'].value_counts()
    rev_xl = rev_xl[rev_xl['userContrib'] > 0.1]
    print(rev_xl.head(5))

    trust = TrustScore([rev_xl['userContrib'], 100]).calculate()

    plotGraph(rev_xl['timestamp'], trust)
#    rr.to_csv("readscore/data-3188459.csv")

    # rr = rev_xl.groupby('userID')['userID'].value_counts()
    # rr.to_csv("readscore/user-count.csv")
    # rev = rev_xl.groupby('userID')['pageid'].value_counts()
    # rev.to_csv("readscore/user-contrib-count.csv")
