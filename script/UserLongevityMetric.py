import json
import sys

import dateparser
import numpy as np

from script.TrustScore import TrustScore
from maya.nltk import util

sys.path.append("..")
from mwapi import Session
from maya.extractors import api
import pandas as pd
from matplotlib import pyplot

session = Session("https://en.wikipedia.org/w/api.php", user_agent="test")
api_extractor = api.Extractor(session)


###### get all users ########
def getAllUsers():
    uc = api_extractor.get_all_user();
    print(uc)
    data = json.dumps(uc)
    df = pd.read_json(data)
    print(df.head())
    df.to_csv("readscore/all_user_data.csv")


###### structure user revision ########
def getUserContrib(userid):
    uc = api_extractor.get_all_contrib_user(userid, {'ids', 'timestamp', 'size'});
    print(uc)

    rev_data = []
    for temp in uc:
        for contrib in temp:
            print(contrib)
            if contrib['parentid'] == 0:
                values = api_extractor.get_all_revision_of_page_prop(contrib['pageid'],
                                                                     rvprop={'ids', 'timestamp', 'userid', 'content'},
                                                                     rv_limit=25, rvstartid=contrib['revid'],
                                                                     should_continue=False)
            else:
                values = api_extractor.get_all_revision_of_page_prop(contrib['pageid'],
                                                                     rvprop={'ids', 'timestamp', 'userid', 'content'},
                                                                     rv_limit=25, rvstartid=contrib['parentid'],
                                                                     should_continue=False)
            values = values[0]
            for id in values:
                try:
                    with open('rev_user/' + str(id['revid']), 'w') as outfile:
                        outfile.write(id['slots']['main']["*"])
                except:
                    print("erreo: ", values)

            values.pop(0)
            if contrib['parentid'] > 0:
                values.pop(0)

            for d in values:
                del d['slots']
            contrib['next_rev'] = values
            rev_data.append(contrib)

    print(rev_data)
    with open('user_data/rev_list_' + userid + '.json', 'w') as outfile:
        json.dump(rev_data, outfile)


###### structure user revision ########
def getUserContribLast(userid):
    with open('user_data/rev_list_' + userid + '.json', 'r') as infile:
        updated_data = json.loads(infile.read())

    for contrib in updated_data:
        print(contrib)

        values = api_extractor.get_all_revision_of_page_prop(contrib['pageid'],
                                                             rvprop={'ids', 'timestamp', 'userid', 'content'},
                                                             rv_limit=1, rv_dir='older',
                                                             should_continue=False)
        values = values[0][0]
        with open('rev_user/' + str(values['revid']), 'w') as outfile:
            outfile.write(values['slots']['main']["*"])

        contrib['last_rev_id'] = values['revid']

    with open('user_data/rev_list_' + userid + '-col.json', 'w') as outfile:
        json.dump(updated_data, outfile)


######## organize list #########
def organizeData(userid):
    with open('user_data/rev_list_' + userid + '.json', 'r') as infile:
        data = json.loads(infile.read())

    page_id = -1
    parent_rev = -1
    size = 0
    rev_id = -1
    count = 0
    updated_data = []
    for temp in data:
        if page_id == -1:
            page_id = temp['pageid']
            parent_rev = temp['parentid']
            rev_id = temp['revid']
            size = temp['size']
        elif page_id == temp['pageid'] and temp['parentid'] == rev_id:
            rev_id = temp['revid']
            temp['parentid'] = parent_rev
            size += temp['size']
            temp['size'] = size
        else:
            page_id = temp['pageid']
            parent_rev = temp['parentid']
            rev_id = temp['revid']
            size = temp['size']
            updated_data.append(data[count - 1])

        count += 1

    print(updated_data)
    with open('user_data/rev_list_' + userid + '-o.json', 'w') as outfile:
        json.dump(updated_data, outfile)


########## calc diff ###########
def calcDiff(userid):
    with open('user_data/rev_list_' + userid + '-o.json', 'r') as infile:
        updated_data = json.loads(infile.read())

    for temp in updated_data:
        captureLongevity = True
        current_rev = util.read_file('rev_user/' + str(temp['revid']))

        if temp['parentid'] == 0:
            original_text = current_rev
        else:
            parent_rev = util.read_file('rev_user/' + str(temp['parentid']))
            original_text = util.findDiffRevised(parent_rev, current_rev)
            original_text = list(v[1] for v in original_text)
            original_text = [w for w in original_text if len(w) > 0]
            small_text = [w for w in original_text if len(w) < 5]



            total = 0
            for txt in original_text:
                total += len(txt)

            temp['contribLength'] = total
            temp['originaltext'] = original_text
            temp['small_text'] = small_text

            rev = [i for i in temp['next_rev']]
            if total > 0 and len(rev) > 5:
                start_time = dateparser.parse(temp['timestamp'])
                print([temp['pageid'], temp['parentid'], temp['revid'], total])
                index = 0
                for id in rev:
                    try:
                        rev_txt = util.read_file('rev_user/' + str(id['revid']))
                        ratio = util.textPreservedRatio(original_text, rev_txt)
                        if ratio < 0.95 and captureLongevity:
                            end_time = dateparser.parse(id['timestamp'])
                            temp['longevityTime'] = round((end_time - start_time).total_seconds() / 3600, 2)
                            temp['longevityRev'] = index
                            captureLongevity = False
                            break
                        id['matchRatio'] = ratio
                    except:
                        print("file error")
                        index -= 1
                    index += 1
                if captureLongevity:
                    temp['longevityRev'] = index

                # last rev contrib
                # rev_txt = util.read_file('rev_user/' + str(temp['last_rev_id']))
                # ratio = util.textPreservedRatio(original_text, rev_txt, total)
                # temp['matchRatioLast'] = ratio

    with open('user_data/rev_list_' + userid + '-dp.json', 'w') as outfile:
        json.dump(updated_data, outfile)


def plotGraphForLongevity(userid):
    with open('user_data/rev_list_' + userid + '-dp.json', 'r') as infile:
        data = json.loads(infile.read())
    for d in data:
        del d['next_rev']

    graph_for = "longevityRev"

    series = pd.DataFrame(data=data)
    series = series[['pageid', 'timestamp', graph_for]]
    series = series[series.longevityRev >= 0]
    series = series.head(70)

    print(series)

    plot = pyplot.plot(series['timestamp'], series[graph_for], 'b-o')

    pyplot.xticks(rotation=45, ha='right')
    pyplot.xlabel("Timestamp")
    pyplot.ylabel("Longevity (No. of Revisions)")
    ax = pyplot.gca()

    pyplot.yticks(np.arange(0, 24 + 1, 2.0))

    ax.set_xticklabels([])
    pyplot.show()


def plotGraphTrustScore(userid):
    with open('user_data/rev_list_' + userid + '-dp.json', 'r') as infile:
        data = json.loads(infile.read())
    for d in data:
        del d['next_rev']

    graph_for = "longevityRev"

    series = pd.DataFrame(data=data)
    series = series[['pageid', 'timestamp', graph_for]]
    series = series[series.longevityRev >= 0]
    series = series.head(70)

    print(series)

    trust_values = TrustScore([series[graph_for], 24]).calculate()

    plot = pyplot.plot(series['timestamp'], trust_values, 'b-o')

    pyplot.xticks(rotation=45, ha='right')
    pyplot.xlabel("Timestamp")
    pyplot.ylabel("Trust Score")
    ax = pyplot.gca()

    ax.set_xticklabels([])
    pyplot.show()


def getPlainText(pageID):
    txt = api_extractor.get_plaintext([pageID])
    with open('rev_user/' + str(id), 'w') as outfile:
        outfile.write(txt['query']['pages'][pageID]['extract'])


def testExtractOriginalContribution():
    source = "abc ghi mno"
    destination = "abc def ghi jkl mno"

    ratio = util.findDiffRevised(source, destination)
    print(ratio)


def testDiffOfContributions():
    parent_rev = [
        "I think the article could  widfdfdth a review.\nFrom memory dfdfdidn't one of our pilots get some dirty US looks for canceling a mission when he decided he couldn't reliably isolate the intended target, as per his Aust. orders accuracy in avoiding civilians had top priority.",
        "I guess you are right. of"]
    current_rev = util.read_file('rev_user/22272908')

    ratio = util.textPreservedRatio(parent_rev, current_rev)
    print(ratio)


if __name__ == "__main__":
    userid = '36440187'  # spammer
    # "userid": 39180130,  commit vandal once
    # "userid": 415269,  good user

    # code to fetch revision of a users, organize them and calculate longevity.
    # Uncomment if you wnat to do it for a new user

    # getUserContrib(userid)
    # getUserContribLast(userid)
    # organizeData(userid)
    #calcDiff(userid)

    # plotGraphForLongevity(userid)
    # plotGraphTrustScore(userid)

    # getAllUsers()

    #test cases
    #testExtractOriginalContribution()
    testDiffOfContributions()

