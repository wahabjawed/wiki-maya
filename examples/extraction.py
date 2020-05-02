import difflib
import json
import sys
from itertools import islice
import dateparser
from dateutil.parser import parser

from maya.nltk import util

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
                values = api_extractor.get_all_revision_of_page_prop(pageid, rvprop={'ids', 'timestamp', 'size', 'userid'})
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
                print("error page: ",temp['pageid'])
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


######## organize list #########
def organizeData(userid):
    with open('rev_user/data/rev_list_' + userid + '.json', 'r') as infile:
        data = json.loads(infile.read())

    # print(data)

    page_id = -1
    parent_rev = -1
    rev_id = -1
    count = 0
    updated_data = []
    for temp in data:
        if page_id == -1:
            page_id = temp['pageid']
            parent_rev = temp['parentid']
            rev_id = temp['revid']
        elif page_id == temp['pageid'] and temp['parentid'] == rev_id:
            rev_id = temp['revid']
            temp['parentid'] = parent_rev
        else:
            page_id = temp['pageid']
            parent_rev = temp['parentid']
            rev_id = temp['revid']
            updated_data.append(data[count - 1])

        count += 1

    print(updated_data)
    with open('rev_user/data/rev_list_' + userid + '-o.json', 'w') as outfile:
        json.dump(updated_data, outfile)


########## calc diff ###########
def calcDiff(userid):
    with open('rev_user/data/rev_list_' + userid + '-c.json', 'r') as infile:
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

    with open('rev_user/data/rev_list_' + userid + '-d.json', 'w') as outfile:
        json.dump(updated_data, outfile)


def plotGraph(userid):
    with open('rev_user/rev_list_' + userid + '-d.json', 'r') as infile:
        data = json.loads(infile.read())
    for d in data:
        del d['next_rev']

    series = pd.DataFrame(data=data)
    series = series[['timestamp', 'longevity']]
    series = series[series.longevity > 0]
    series.plot()
    pyplot.show()

    pyplot.plot(series['timestamp'], series['longevity'])
    pyplot.show()


def getPlainText(pageID):
    txt = api_extractor.get_plaintext([pageID])
    with open('rev_user/' + str(id), 'w') as outfile:
        outfile.write(txt['query']['pages'][pageID]['extract'])


if __name__ == "__main__":
    # getUserContrib('5834659')
    # organizeData('5834659')
    getUserContribPercentage(5834659)
    #calcDiff('5834659')
