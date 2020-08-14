import json
import sys

import dateparser
import numpy as np

from maya.nltk import util
from pandarallel import pandarallel

from script.TrustScore import TrustScore

sys.path.append("..")
from mwapi import Session
from maya.extractors.api import Extractor
import pandas as pd
from matplotlib import pyplot

import os.path

session = Session("https://en.wikipedia.org/w/api.php", user_agent="test")
api_extractor = Extractor(session)

pandarallel.initialize(nb_workers=5)


def getAllUsers():
    """
      Fetches the list of all the user in Wikipedia that has contributed.
      Args:

      Result:
          the dataframe of the list of users and their editcounts
       """
    uc = api_extractor.get_all_user();
    print(uc)
    data = json.dumps(uc)
    df = pd.read_json(data)
    print(df.head())
    df.to_csv("readscore/all_user_data.csv")


def getUserContrib(user_id):
    """
      Fetches all the contribution of the user[revision] and for each contribution, it fetches the next 50 revisions after it.
      it also save the content of each revision on the disk named as the 'revision number'
      Args:
          user_id (str): user id of the user.
      Result:
          the list of revisions contributed by user and the for each revision it has the list of next 50 revisions.
       """
    user_contrib = api_extractor.get_all_contrib_user(user_id, {'ids', 'timestamp', 'size'});
    print(user_contrib)

    rev_data = []
    for row in user_contrib:
        for item_contrib in row:
            print(item_contrib)
            if item_contrib['parentid'] == 0:
                values = api_extractor.get_all_revision_of_page_prop(item_contrib['pageid'],
                                                                     rvprop={'ids', 'timestamp', 'userid', 'content'},
                                                                     rv_limit=50, rvstartid=item_contrib['revid'],
                                                                     should_continue=True, continue_until=2)
            else:
                values = api_extractor.get_all_revision_of_page_prop(item_contrib['pageid'],
                                                                     rvprop={'ids', 'timestamp', 'userid', 'content'},
                                                                     rv_limit=50, rvstartid=item_contrib['parentid'],
                                                                     should_continue=True, continue_until=2)
            for id in values:
                try:
                    with open('rev_user/' + str(id['revid']), 'w') as outfile:
                        outfile.write(id['slots']['main']["*"])
                except:
                    print("error: ", values)

            values.pop(0)
            if item_contrib['parentid'] > 0:
                values.pop(0)

            for d in values:
                del d['slots']
            item_contrib['next_rev'] = values
            rev_data.append(item_contrib)

    print(rev_data)
    with open('user_data/rev_list_' + user_id + '.json', 'w') as outfile:
        json.dump(rev_data, outfile)


def getUserContribLast(userid):
    """
      Fetches the last/latest revision of each revision on each page committed by the user_id
      it also save the content of each revision on the disk named as the 'revision number'
      Args:
          user_id (str): user id of the user.
      Result:
          the list of last/latest revision on each page on which the user_id contributed.
       """
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


def organizeData(userid):
    """
      It organizes the data by combining the consecutive revision of the same user on same page w.r.t time into a single revision.
      Args:
          user_id (str): user id of the user.
      Result:
          the list of revisions contributed by user and the for each revision it has the list of next 50 revisions.
       """
    with open('user_data/rev_list_' + userid + '.json', 'r') as infile:
        data = json.loads(infile.read())

    page_id = -1
    parent_rev = -1
    size = 0
    rev_id = -1
    count = 0
    updated_data = []
    for row in data:
        if page_id == -1:
            page_id = row['pageid']
            parent_rev = row['parentid']
            rev_id = row['revid']
            size = row['size']
        elif page_id == row['pageid'] and row['parentid'] == rev_id:
            rev_id = row['revid']
            row['parentid'] = parent_rev
            size += row['size']
            row['size'] = size
        else:
            page_id = row['pageid']
            parent_rev = row['parentid']
            rev_id = row['revid']
            size = row['size']
            updated_data.append(data[count - 1])

        count += 1

    print(updated_data)
    with open('user_data/rev_list_' + userid + '-o.json', 'w') as outfile:
        json.dump(updated_data, outfile)


def calcDiff(user_id):
    """
      It calculates the longevity of the contribution of user in the next 50 revision
      Args:
          user_id (str): user id of the user.
      Result:
          the list of revisions contributed by user and the for each revision it has the Longevity value in no of revision and time.
       """
    try:
        with open('user_data/rev_list_' + user_id + '-o.json', 'r') as infile:
            updated_data = json.loads(infile.read())

        for row in updated_data:
            print("Picking For Analysis Artcile,Parent,Revision: ", [row['pageid'], row['parentid'], row['revid']])
            capture_longevity = True
            current_rev = util.read_file('rev_user/' + str(row['revid']))

            if row['parentid'] == 0:
                original_text = current_rev
            else:
                parent_rev = util.read_file('rev_user/' + str(row['parentid']))
                original_text = util.findDiffRevised(parent_rev, current_rev)
                original_text = list(v[1] for v in original_text)
                original_text = [w for w in original_text if len(w) > 1]
                small_text = [w for w in original_text if len(w) < 5]

                total = 0
                for txt in original_text:
                    total += len(txt)

                row['contribLength'] = total
                row['originaltext'] = original_text
                row['small_text'] = small_text

                next_revs = [i for i in row['next_rev']]
                if total > 0:
                    start_time = dateparser.parse(row['timestamp'])
                    print("Performing Diff For Artcile,Parent,Revision: ",
                          [row['pageid'], row['parentid'], row['revid'], total])
                    index = 0
                    for rev in next_revs:
                        try:
                            next_rev = util.read_file('rev_user/' + str(rev['revid']))
                            d_text = util.getInsertedContentSinceParentRevision(parent_rev, next_rev)
                            ratio = util.textPreservedRatioStrict(original_text, d_text)
                            print("ratio: ", ratio)
                            if ratio < 0.95 and capture_longevity:
                                end_time = dateparser.parse(rev['timestamp'])
                                row['longevityTime'] = round((end_time - start_time).total_seconds() / 3600, 2)
                                row['longevityRev'] = index
                                row['matchRatio'] = ratio
                                capture_longevity = False
                                print("longevity-S: ", index)
                                break
                        except Exception as e:
                            print("file error", e.message)
                            index -= 1
                        index += 1
                    if capture_longevity:
                        row['longevityRev'] = index
                        end_time = dateparser.parse(rev['timestamp'])
                        row['longevityTime'] = round((end_time - start_time).total_seconds() / 3600, 2)
                        row['matchRatio'] = ratio
                        print("longevity-L: ", index)
        if len(updated_data) > 0:
            with open('user_data_50_90_b_1/rev_list_' + user_id + '-dp.json', 'w') as outfile:
                json.dump(updated_data, outfile)
    except Exception as e:
        print("skipping diff as no contribution: ", e)


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


def testDiffOfContributionStrict():
    parent_rev = [
        "I think the article could  widfdfdth a review.\nFrom memory dfdfdidn't one of our pilots get some dirty US looks for canceling a mission when he decided he couldn't reliably isolate the intended target, as per his Aust. orders accuracy in avoiding civilians had top priority.",
        "Thanks Cunch. I a guess you are ."]
    current_rev = util.read_file('22272908')

    ratio = util.textPreservedRatioStrict([parent_rev[1]], current_rev)
    print(ratio)


def processData(row):
    if row['status'] == 1:
        index = row[0]
        print(row)
        print("Index: " + str(index))
        userid = str(row['id'])
        #         if getUserContrib(userid) >0:
        #             organizeData(userid)
        calcDiff(userid)
        #             user_data.iloc[index, 4:5] = 1
        #         else:
        #              user_data.iloc[index, 4:5] = 2

        print("Saving Status for User ", user_data.iloc[index, :])
        # user_data.to_csv("all_user_data_c.csv")


def updateStatusInCSVForDiff():
    for row in user_data.iterrows():
        ids = str(row[1]['id'])
        path = 'user_data_50_90_s/rev_list_' + ids + '-dp.json'
        print(path)
        if os.path.isfile(path) == 1:
            user_data.iloc[row[0], 4:5] = 1
            print('Exist')

    user_data.to_csv("csv/all_user_data_c_50_90_s.csv")


if __name__ == "__main__":
    userid = '415269'  # spammer
    # "userid": 39180130,  commit vandal once
    # "userid": 415269,  good user

    # code to fetch revision of a users, organize them and calculate longevity.
    # Uncomment if you wnat to do it for a new user

    # getUserContrib(userid)
    # getUserContribLast(userid)
    # organizeData(userid)
    calcDiff(userid)

    # plotGraphForLongevity(userid)
    # plotGraphTrustScore(userid)

    # getAllUsers()

#    user_data = pd.read_csv("csv/all_user_data_c_50_90_s.csv")
    # user_data.apply(processData, axis=1)
    # updateStatusInCSVForDiff()
