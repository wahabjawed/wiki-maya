import glob
import multiprocessing
import os
from itertools import islice

import language_check
import pandas as pd
from textstat import textstat
# import textstat

from maya.nltk import util
from readcalc import readcalc
from mwapi import Session
from maya.extractors import api


def calcFeatures(index, params):
    rev = params  # Multiprocessing...
    global rev_xl
    filename = "/Users/abdulwahab/Desktop/internship/dataset/2015_english_wikipedia_quality_dataset/revisiondata/" + str(
        rev['revid'])
    if (os.path.exists(filename)):
        text = util.read_file(filename)
        text = util.cleanhtml(text)
        text = text.replace('\'\'\'', '')
        assert rev['pageid'] == rev_xl.iloc[index, 0]
        print("matched ", rev['revid'])

        gra = len(tool.check(text))

        # try:
        #     gras = (gra / len(util.sent_tokenize(text)))
        # except:
        #     gras = 0
        #
        # try:
        #     graw = (gra / len(util.word_tokenize(text))) * 100
        # except:
        #     graw = 0

        #tt = gra

        print(rev['revid'], gra)

    rev_xl.iloc[index, 36:37] = gra

    return rev_xl.iloc[index, :]


def calcQuality(index, params):
    rev = params  # Multiprocessing...
    global rev_xl

    tt = 0.0

    if rev['rating'] == 'Stub':
        tt = 0.0
    elif rev['rating'] == 'Start':
        tt = 0.2
    elif rev['rating'] == 'C':
        tt = 0.4
    elif rev['rating'] == 'B':
        tt = 0.6
    elif rev['rating'] == 'GA':
        tt = 0.8
    elif rev['rating'] == 'FA':
        tt = 1.0

    rev_xl.iloc[index, 34:35] = tt

    return rev_xl.iloc[index, :]


def startCalcFeatures():
    global rev_xl

    new_column = ['grammar']

    rev_xl = rev_xl.reindex(columns=rev_xl.columns.tolist() + new_column)

    # Perform classification.
    for index, row in rev_xl.iterrows():
        calcFeatures(index, row)

    rev_xl.to_csv("readscore/all_score_train-c-output.csv")


if __name__ == "__main__":
    rev_xl = pd.read_csv("readscore/all_score_train-c-output.csv",
                         dtype={0: 'int32', 1: 'int32', 2: 'object'})
    tool = language_check.LanguageTool('en-US')
    startCalcFeatures()
