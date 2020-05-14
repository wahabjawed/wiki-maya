import json
import sys
from itertools import islice
import dateparser
from dateutil.parser import parser
from textstat import textstat
from gensim.parsing.preprocessing import remove_stopwords
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
                with open('rev_user/' + str(id['revid']), 'w') as outfile:
                    outfile.write(id['slots']['main']["*"])

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


###### structure user revision ########
def getUserContribLast(userid):
    with open('rev_user/data/rev_list_' + userid + '-co.json', 'r') as infile:
        updated_data = json.loads(infile.read())

    for contrib in updated_data:
        # d= dict(filter(lambda i:i[0] in ['pageid','revid','parentid'], contrib.items()()))
        print(contrib)

        values = api_extractor.get_all_revision_of_page_prop(contrib['pageid'],
                                                             rvprop={'ids', 'timestamp', 'userid', 'content'},
                                                             rv_limit=1, rv_dir='older',
                                                             should_continue=False)
        values = values[0][0]
        with open('rev_user/' + str(values['revid']), 'w') as outfile:
            outfile.write(values['slots']['main']["*"])

        contrib['last_rev_id'] = values['revid']

    with open('rev_user/data/rev_list_' + userid + '-col.json', 'w') as outfile:
        json.dump(updated_data, outfile)

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
    with open('rev_user/data/rev_list_' + userid + '-col.json', 'r') as infile:
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

            if total > 5:
                start_time = dateparser.parse(temp['timestamp'])
                print([temp['pageid'], temp['parentid'], temp['revid'], total])
                rev = [i for i in temp['next_rev']]
                for id in rev:
                    rev_txt = util.read_file('rev_user/' + str(id['revid']))
                    ratio = util.textPreservedRatio(original_text, rev_txt, total)
                    if ratio < 0.7 and captureLongevity:
                        end_time = dateparser.parse(id['timestamp'])
                        temp['longevity'] = round((end_time - start_time).total_seconds() / 3600, 2)
                        captureLongevity = False
                        # break
                    id['matchRatio'] = ratio
                #last rev contrib
                rev_txt = util.read_file('rev_user/' + str(temp['last_rev_id']))
                ratio = util.textPreservedRatio(original_text, rev_txt, total)
                temp['matchRatioLast'] = ratio

    with open('rev_user/data/rev_list_' + userid + '-dp.json', 'w') as outfile:
        json.dump(updated_data, outfile)


def plotGraph(userid):
    with open('rev_user/data/rev_list_' + userid + '-dp.json', 'r') as infile:
        data = json.loads(infile.read())
    for d in data:
        del d['next_rev']

    graph_for = "matchRatioLast"

    series = pd.DataFrame(data=data)
    series = series[['pageid', 'timestamp', graph_for]]
    series = series[series.matchRatioLast > 0]

    print(series)

    plot = pyplot.plot(series['timestamp'], series[graph_for], 'b-o')

    for row in series.iterrows():
        if row[1][graph_for] > 1000:
            pyplot.annotate(row[1]['pageid'], (row[1]['timestamp'], row[1][graph_for]))

    pyplot.xticks(rotation=45, ha='right')
    pyplot.xlabel("Timestamp")
    pyplot.ylabel("Longevity ( Mins)")
    pyplot.show()


def plotGraphFour(userid):
    with open('rev_user/data/rev_list_' + userid + '-dp.json', 'r') as infile:
        data = json.loads(infile.read())
    for d in data:
        del d['next_rev']

    graph_for = "size"

    series = pd.DataFrame(data=data)
    series = series[['pageid', 'timestamp', graph_for]]
    series = series[series['size'] > 4000]

    plot = pyplot.plot(series['timestamp'], series[graph_for], 'b-o')

    # for row in series.iterrows():
    #     if row[1][graph_for] > 70000:
    #         pyplot.annotate(row[1]['pageid'], (row[1]['timestamp'], row[1][graph_for]))

    pyplot.xticks(rotation=45, ha='right')
    pyplot.xlabel("Timestamp")
    pyplot.ylabel("User Contribution (size)")
    pyplot.show()


def plotGraphThree(userid):
    with open('rev_user/data/rev_list_' + userid + '-dp.json', 'r') as infile:
        data = json.loads(infile.read())
    for d in data:
        del d['next_rev']

    graph_for = "longevity"

    series = pd.DataFrame(data=data)
    series = series[['userContrib', 'timestamp', graph_for]]
    series = series[series.longevity > 0]

    ax = pyplot.axes(projection='3d')

    # Data for three-dimensional scattered points
    xdata = series['timestamp']
    zdata = series['userContrib']
    ydata = series[graph_for]
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

    pyplot.xlabel("Timestamp")
    pyplot.ylabel("Longevity ( Mins)")
    pyplot.show()


def plotGraphTwo(userid):
    with open('rev_user/data/rev_list_' + userid + '-dp.json', 'r') as infile:
        data = json.loads(infile.read())
    for d in data:
        if "longevity" in d.keys():
            toPlot = d['next_rev']
            print("plotting for page: ", d['pageid'])
            mr = [round(mr['matchRatio'] * 100, 2) for mr in toPlot]
            ts = [ts['timestamp'] for ts in toPlot]
            pyplot.plot(ts, mr, label=d['pageid'])

    pyplot.xlabel("Timestamp", rotation=90)
    pyplot.ylabel("Longevity")
    pyplot.show()


def getPlainText(pageID):
    txt = api_extractor.get_plaintext([pageID])
    with open('rev_user/' + str(id), 'w') as outfile:
        outfile.write(txt['query']['pages'][pageID]['extract'])


def testFeature():
    rev_text = []
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    rev_files = glob.glob("/Users/abdulwahab/Desktop/internship/wikipedia_analysis-master/lang_model/enwiki/text/*")
    for file in rev_files:
        file_name = os.path.basename(file)
        rev_text.append((file_name, file))

    rev_xl = pd.read_csv("/Users/abdulwahab/Desktop/internship/wikipedia_analysis-master/analysis/all_score_train.csv",
                         dtype={0: 'int32', 1: 'int32', 2: 'object'})

    for index,row in rev_xl.iterrows():
        filename = "/Users/abdulwahab/Desktop/internship/dataset/2015_english_wikipedia_quality_dataset/revisiondata/" + str(row['revid'])
        if (os.path.exists(filename)):
            print(str(row['revid']))
            text = util.read_file(filename)
            filtered_sentence = text
            #filtered_sentence = remove_stopwords(text)

            #print(filtered_sentence)
            result = [textstat.flesch_reading_ease(filtered_sentence), textstat.flesch_kincaid_grade(filtered_sentence), textstat.smog_index(filtered_sentence),
                      textstat.coleman_liau_index(filtered_sentence)]

            # print(result)
            #
            # filtered_sentence = util.cleanhtml(filtered_sentence)
            # result = [textstat.flesch_reading_ease(filtered_sentence), textstat.flesch_kincaid_grade(filtered_sentence), textstat.smog_index(filtered_sentence),
            #           textstat.coleman_liau_index(filtered_sentence)]

            print(result)

if __name__ == "__main__":
    userid = '5834659'
    #getUserContribLast(userid)
    #organizeData(userid)
    # getUserContribPercentage(5834659)
    #calcDiff(userid)
    # organizeDataTwo('5834659')
    #plotGraphFour(userid)
    #plotGraph(userid)
    testFeature()



