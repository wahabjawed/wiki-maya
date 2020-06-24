import numpy
import pandas as pd
from mwapi import Session

from maya.extractors import api



def calcQuality(params):
    rev = params

    if rev['rating'] == 'Stub':
        quality_score = 0.1
    elif rev['rating'] == 'Start':
        quality_score = 0.28
    elif rev['rating'] == 'C':
        quality_score = 0.46
    elif rev['rating'] == 'B':
        quality_score = 0.64
    elif rev['rating'] == 'GA':
        quality_score = 0.82
    elif rev['rating'] == 'FA':
        quality_score = 1.0

    return quality_score


def calcFeatures(index, params):
    rev = params
    global rev_xl
    global existingContrib
    print(rev['revid'], '---', rev['userContrib'])

    if numpy.isnan(rev['userContrib']):

        test = api_extractor.get_rev_doc_map([rev['revid']], {'ids', 'user', 'userid', 'timestamp'})

        print(test)
        try:
            pageId = test[rev['revid']]['page']['pageid']
            userid = test[rev['revid']]['userid']
            timestamp = test[rev['revid']]['timestamp']
            assert pageId == rev_xl.iloc[index, 0]

            pageid = rev['pageid']
            print("page ID:", pageid, '--', index)
            if pageid not in existingContrib:
                values = api_extractor.get_all_revision_of_page_prop(pageid,
                                                                     rvprop={'ids', 'timestamp', 'size', 'userid'})
                allUserContrib = dict()
                totalContrib = 0
                for item in values:
                    for revision in item:
                        if 'userid' in revision.keys():
                            totalContrib += revision['size']
                            if (revision['userid'] in allUserContrib):
                                allUserContrib[revision['userid']] += revision['size']
                            else:
                                allUserContrib[revision['userid']] = revision['size']

                userContrib = round(allUserContrib[userid] / totalContrib * 100, 2)
                print(userContrib)
                re = userContrib
                existingContrib[pageid] = userContrib

            else:
                print("picking existing contrib:", existingContrib[pageid])
                re = existingContrib[pageid]
        except:
            re = 0.0
        quality_score = calcQuality(rev)
        contribQuality = round(quality_score * re, 2)
        rev_xl.iloc[index, 35:40] = [quality_score, re, str(userid), str(timestamp), contribQuality]

        print(rev_xl.iloc[index, :])

        if index % 10 == 0:
            rev_xl.to_csv(path)


def startCalcFeatures():
    global rev_xl

    new_column = ['quality_score', 'userContrib', 'userID', 'timestamp', 'contribQuality']

    rev_xl = rev_xl.reindex(columns=rev_xl.columns.tolist() + new_column)

    # Perform classification.
    for index, row in rev_xl.iterrows():
        calcFeatures(index, row)

    rev_xl.to_csv(path)


if __name__ == "__main__":
    path = "readscore/all_score_train-c-output.csv"

    rev_xl = pd.read_csv(path,
                         dtype={0: 'int32', 1: 'int32', 2: 'object'})
    session = Session("https://en.wikipedia.org/w/api.php", user_agent="test")
    api_extractor = api.Extractor(session)
    existingContrib = dict()
    startCalcFeatures()
