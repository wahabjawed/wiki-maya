import json
import sys

import numpy as np

from examples.TrustScore import TrustScore

sys.path.append("..")
import pandas as pd
from matplotlib import pyplot


def plotGraph(x, y,label):
    plot = pyplot.plot(x, y, 'b-')

    pyplot.xticks(rotation=45, ha='right')
    pyplot.xlabel("Timestamp")
    pyplot.ylabel(label)
    ax = pyplot.gca()

    ax.set_xticklabels([])
    pyplot.show()


if __name__ == "__main__":
    userid = 3188459
    # "userid": 39180130

    rev_xl = pd.read_csv("readscore/data.csv",
                         dtype={0: 'int32', 1: 'int32', 2: 'object', 6: 'int32'})

    rev_xl = rev_xl[rev_xl['userID'] == userid]
    rev_xl = rev_xl[rev_xl['userContrib'] > 0.1]
    print(rev_xl.head(5))

    trust_score = TrustScore([rev_xl['userContrib'], 100]).calculate()
    trust_score_quality = TrustScore([rev_xl['contribQuality'], 100]).calculate()

    plotGraph(rev_xl['timestamp'], rev_xl['userContrib'], "User Contribution")
    plotGraph(rev_xl['timestamp'], trust_score, "Trust Score (Contribution)")
    plotGraph(rev_xl['timestamp'], rev_xl['contribQuality'], "Quality of Contribution")
    plotGraph(rev_xl['timestamp'], trust_score_quality, "Trust Score (Quality of Contribution)")
