import json
import sys
from itertools import islice
import dateparser
from dateutil.parser import parser
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from textstat import textstat
from gensim.parsing.preprocessing import remove_stopwords

from examples.TrustScore import TrustScore
from maya.nltk import util
import numpy as np
import glob
import multiprocessing
import os

sys.path.append("..")
from mwapi import Session
from maya.extractors import api
import pandas as pd
from matplotlib import pyplot
from maya.features import temporal, wikitext, bytes

session = Session("https://en.wikipedia.org/w/api.php", user_agent="test")
api_extractor = api.Extractor(session)


# features = [temporal.revision.day_of_week,
#             temporal.revision.hour_of_day,
#             wikitext.revision.content_chars,
#             wikitext.revision.headings,
#             wikitext.revision.tags,
#             wikitext.revision.wikilinks,
#             wikitext.revision.punctuations,
#             wikitext.revision.external_links,
#             wikitext.revision.longest_repeated_char,
#             bytes.revision.length,
#             bytes.revision.parent.length,
#             wikitext.revision.parent.headings_by_level(2)]
#
# values = api_extractor.extract(944941487, features)
# for feature, value in zip(features, values):
#     print("\t{0}: {1}".format(feature, repr(value)))


# test = api_extractor.get_rev_doc_map([944941487])
#
# print(test)
# pageId = test[944941487]['page']['pageid']
# userId = test[944941487]['userid']
# text = test[944941487]['slots']['main']['*']
#
#
# test1 = api_extractor.get_plaintext([944941487])

def getUserContribPercentage(userid):
    with open('rev_user/data/rev_list_' + str(userid) + '-c.json', 'r') as infile:
        updated_data = json.loads(infile.read())
    existingContrib = dict()
    for temp in updated_data:
        if ('userContrib' not in temp.keys()):
            pageid = temp['pageid']
            print("page ID:", pageid)
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
                temp['userContrib'] = userContrib
                existingContrib[pageid] = userContrib
                with open('rev_user/data/rev_list_' + str(userid) + '-c.json', 'w') as outfile:
                    json.dump(updated_data, outfile)
            else:
                print("picking existing contrib:", existingContrib[pageid])
                temp['userContrib'] = existingContrib[pageid]
                with open('rev_user/data/rev_list_' + str(userid) + '-c.json', 'w') as outfile:
                    json.dump(updated_data, outfile)
        else:
            try:
                existingContrib[temp['pageid']] = temp['userContrib']
            except:
                print("error page: ", temp['pageid'])
            print("skipped page ID:", temp['pageid'])


# ctext = util.cleanhtml(text)
# grammar_rate = util.check_grammar_error_rate(ctext)
# fog = util.gunning_fog(ctext)
# print(fog)


###### structure user revision ########
def getUserContrib(userid):
    uc = api_extractor.get_all_contrib_user(userid, {'ids', 'timestamp', 'size'});
    print(uc)

    rev_data = []
    for temp in uc:
        for contrib in temp:
            # d= dict(filter(lambda i:i[0] in ['pageid','revid','parentid'], contrib.items()()))
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
                    print("erreo: ", id)

            values.pop(0)
            if contrib['parentid'] > 0:
                values.pop(0)

            for d in values:
                del d['slots']
            contrib['next_rev'] = values
            rev_data.append(contrib)

    print(rev_data)
    with open('rev_user/data/rev_list_' + userid + '.json', 'w') as outfile:
        json.dump(rev_data, outfile)


######## organize list #########
def organizeData(userid):
    with open('rev_user/data/rev_list_' + userid + '.json', 'r') as infile:
        data = json.loads(infile.read())

    # print(data)

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
    with open('rev_user/data/rev_list_' + userid + '-o.json', 'w') as outfile:
        json.dump(updated_data, outfile)


######## organize list Two #########
def organizeDataTwo(userid):
    with open('rev_user/data/rev_list_' + str(userid) + '-c.json', 'r') as infile:
        data = json.loads(infile.read())

    for itrOne in data:
        revid = itrOne['revid']
        parentid = itrOne['parentid']
        pageid = itrOne['pageid']
        indOne = data.index(itrOne)
        for itrTwo in data:
            if itrTwo['pageid'] == pageid and itrTwo['parentid'] == revid:
                print("found :", itrTwo['pageid'], " ", itrTwo['parentid'])
                itrTwo['parentid'] = parentid
                revid = itrTwo['revid']
                print(data[indOne])
                del data[indOne]
                indOne = data.index(itrTwo)

    print(data)
    with open('rev_user/data/rev_list_' + userid + '-co.json', 'w') as outfile:
        json.dump(data, outfile)


########## calc diff ###########
def calcDiff(userid):
    with open('rev_user/data/rev_list_' + userid + '-o.json', 'r') as infile:
        updated_data = json.loads(infile.read())

    for temp in updated_data:
        rev = []
        captureLongevity = True
        # rev.append(temp['parentid'])
        # rev.append(temp['revid'])

        current_rev = util.read_file('rev_user/' + str(temp['revid']))

        if temp['parentid'] == 0:
            original_text = current_rev
        else:
            parent_rev = util.read_file('rev_user/' + str(temp['parentid']))
            original_text = util.findDiffRevised(parent_rev, current_rev)
            original_text = [w for w in original_text if len(w[1]) > 2]

            total = 0
            for txt in original_text:
                total += len(txt[1])

            temp['contribLength'] = total
            temp['originaltext'] = original_text

            rev = [i for i in temp['next_rev']]
            if total > 10 and len(rev) > 7:
                start_time = dateparser.parse(temp['timestamp'])
                print([temp['pageid'], temp['parentid'], temp['revid'], total])
                index = 0
                for id in rev:
                    try:
                        rev_txt = util.read_file('rev_user/' + str(id['revid']))
                        ratio = util.textPreservedRatio(original_text, rev_txt, total)
                        if ratio < 0.4 and captureLongevity:
                            end_time = dateparser.parse(id['timestamp'])
                            temp['longevity'] = round((end_time - start_time).total_seconds() / 3600, 2)
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

    with open('rev_user/data/rev_list_' + userid + '-dp.json', 'w') as outfile:
        json.dump(updated_data, outfile)


def plotGraph(userid):
    with open('rev_user/data/rev_list_36440187-dp.json', 'r') as infile:
        data = json.loads(infile.read())
    for d in data:
        del d['next_rev']

    graph_for = "longevityRev"

    series1 = pd.DataFrame(data=data)
    series1 = series1[['pageid', 'timestamp', graph_for]]
    series1 = series1[series1.longevityRev >= 0]
    series1['longevityRevP'] = series1['longevityRev'].shift(1)
    y = TrustScore([series1['longevityRev'], 24]).calculate()

    with open('rev_user/data/rev_list_39180130-dp.json', 'r') as infile:
        data = json.loads(infile.read())
    for d in data:
        del d['next_rev']

    series2 = pd.DataFrame(data=data)
    series2 = series2[['pageid', 'timestamp', graph_for]]
    series2 = series2[series2.longevityRev >= 0]
    series2['longevityRevP'] = series2['longevityRev'].shift(1)
    y.extend(TrustScore([series2['longevityRev'], 24]).calculate())

    with open('rev_user/data/rev_list_415269-dp.json', 'r') as infile:
        data = json.loads(infile.read())
    for d in data:
        del d['next_rev']

    series3 = pd.DataFrame(data=data)
    series3 = series3[['pageid', 'timestamp', graph_for]]
    series3 = series3[series3.longevityRev >= 1]
    series3['longevityRevP'] = series3['longevityRev'].shift(1)

    y.extend(TrustScore([series3['longevityRev'], 24]).calculate())

    series1 = series1.append(series2)
    series1 = series1.append(series3)
    series1["longevityRevP"] = series1["longevityRevP"].fillna(0)
    series1["Trust"] = y
    series1['TrustP'] = series1['Trust'].shift(1)
    series1["TrustP"] = series1["TrustP"].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        series1[['Trust', 'longevityRevP', 'TrustP']].values.reshape(-1, 3),
        series1['longevityRev'].values.reshape(-1, 1), test_size=0.10, random_state=3)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # To retrieve the intercept:
    print(regressor.intercept_)
    # For retrieving the slope:
    print(regressor.coef_)

    y_pred = regressor.predict(X_test)

    y_pred = y_pred.reshape(1,-1)[0]
    y_test = y_test.reshape(1,-1)[0]

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('R2 Score:', metrics.r2_score(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    df.plot(kind='bar', figsize=(10, 8))
    pyplot.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    pyplot.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    pyplot.xlabel("Samples")
    pyplot.ylabel("Longevity (Revision)")
    pyplot.show()




def getPlainText(pageID):
    txt = api_extractor.get_plaintext([pageID])
    with open('rev_user/' + str(id), 'w') as outfile:
        outfile.write(txt['query']['pages'][pageID]['extract'])


if __name__ == "__main__":
    # good user Nick-D
    userid = '29047545'

    # 29047545 CPA-5

    # vandal -- blocked  Kujidamanbomb 39116775
    # spammer -- blocked "user": "Zozazaroo", "userid": 36440187,

    # once vandal -- Serols  -- 9929111
    # "user": "XxPixel WarriorxX",
    # "userid": 39180130,  did vandal on dolly

    # getUserContribLast(userid)
    # getUserContrib(userid)
    # organizeData(userid)
    # getUserContribPercentage(5834659)
    # calcDiff(userid)
    # organizeDataTwo('5834659')
    # plotGraphFour(userid)
    plotGraph(userid)
    # testFeature()
