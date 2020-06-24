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
    print(rev['revid'],'---',rev['userContrib'])

    if numpy.isnan(rev['userContrib']):

        test = api_extractor.get_rev_doc_map([rev['revid']],{'ids', 'user','userid'})

        print(test)
        try:
            pageId = test[rev['revid']]['page']['pageid']
            userid = test[rev['revid']]['userid']
            assert pageId == rev_xl.iloc[index, 0]

            pageid = rev['pageid']
            print("page ID:", pageid,'--',index)
            if pageid not in existingContrib:
                values = api_extractor.get_all_revision_of_page_prop(pageid,
                                                                     rvprop={'ids', 'timestamp', 'size', 'userid'})
                allUserContrib = dict()
                totalContrib = 0
                for item in values:
                    for rev in item:
                        if 'userid' in rev.keys():
                            totalContrib += rev['size']
                            if (rev['userid'] in allUserContrib):
                                allUserContrib[rev['userid']] += rev['size']
                            else:
                                allUserContrib[rev['userid']] = rev['size']

                userContrib = round(allUserContrib[userid] / totalContrib * 100, 2)
                print(userContrib)
                re = userContrib
                existingContrib[pageid] = userContrib

            else:
                print("picking existing contrib:", existingContrib[pageid])
                re = existingContrib[pageid]
        except:
            re = 0.0
        rev_xl.iloc[index, 35:36] = re

        if index % 10 == 0:
            rev_xl.to_csv("readscore/all_score_train-c-output.csv")

    return rev_xl.iloc[index, :]




def startCalcFeatures():
    global rev_xl

    #new_column = ['userContrib']

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