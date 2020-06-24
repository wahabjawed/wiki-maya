import glob
import multiprocessing
import os
from itertools import islice

import language_check
import numpy
import pandas as pd
from textstat import textstat
#import textstat

from maya.nltk import util
from readcalc import readcalc
from mwapi import Session
from maya.extractors import api

def calcFeatures(index,params):
    rev = params  # Multiprocessing...
    global rev_xl
    global existingContrib
    print(rev['revid'],'---', index)

    if numpy.isnan(rev['userID']):

        test = api_extractor.get_rev_doc_map([rev['revid']],{'ids', 'user','userid','timestamp'})

        print(test)
        try:
            pageId = test[rev['revid']]['page']['pageid']
            re = [str(test[rev['revid']]['userid']), str(test[rev['revid']]['timestamp'])]
            assert pageId == rev_xl.iloc[index, 0]
        except:
            re = ["0", "0"]
        rev_xl.iloc[index, 37:39] = re

        if index % 10 == 0:
            rev_xl.to_csv("readscore/all_score_train-c-output.csv")

    return rev_xl.iloc[index, :]




def startCalcFeatures():
    global rev_xl

    new_column = ['userID', 'timestamp']

    rev_xl = rev_xl.reindex(columns=rev_xl.columns.tolist())

    # Perform classification.
    for index, row in rev_xl.iterrows():
        calcFeatures(index, row)

    rev_xl.to_csv("readscore/all_score_train-c-output.csv")


if __name__ == "__main__":
    rev_xl = pd.read_csv("readscore/all_score_train-c-output.csv",
                         dtype={0: 'int32', 1: 'int32', 2: 'object'})
    session = Session("https://en.wikipedia.org/w/api.php", user_agent="test")
    api_extractor = api.Extractor(session)
    existingContrib = dict()
    startCalcFeatures()
