import sys

from maya.nltk import util

sys.path.append("..")
from mwapi import Session
from maya.extractors import api
from maya.features import temporal, wikitext

session = Session("https://en.wikipedia.org/w/api.php", user_agent="test")
api_extractor = api.Extractor(session)

# features = [temporal.revision.day_of_week,
#             temporal.revision.hour_of_day,
#             wikitext.revision.content_chars,
#             wikitext.revision.parent.headings_by_level(2)]
#
# values = api_extractor.extract(624577024, features)
# for feature, value in zip(features, values):
#     print("\t{0}: {1}".format(feature, repr(value)))


test = api_extractor.get_rev_doc_map([944941487])

print(test)
pageId = test[944941487]['page']['pageid']
userId = test[944941487]['userid']
text = test[944941487]['slots']['main']['*']

# values = api_extractor.get_all_revision_of_page_prop(pageId, rvprop={'ids', 'timestamp', 'size', 'userid'})
# print(values)
#
# allUserContrib = dict()
# totalContrib=0
# for item in values:
#     for rev in item:
#         totalContrib+=rev['size']
#         if(rev['userid'] in allUserContrib):
#             allUserContrib[rev['userid']] += rev['size']
#         else:
#             allUserContrib[rev['userid']] = rev['size']
#
# userContrib = allUserContrib[userId]/totalContrib*100
# print(userContrib)

ctext = util.cleanhtml(text)
grammar_rate = util.check_grammar_error_rate(ctext)

uc = api_extractor.get_all_contrib_user(userId);
print(uc)
