import glob
import multiprocessing
import os
from itertools import islice

import language_check
import pandas as pd

from maya.nltk import util
from readcalc import readcalc


def calcFeatures(params):
    index, rev = params  # Multiprocessing...
    global rev_xl
    filename = "/Users/abdulwahab/Desktop/internship/dataset/2015_english_wikipedia_quality_dataset/revisiondata/" + str(
        rev['revid'])
    if (os.path.exists(filename)):
        print(rev['revid'])
        text = util.read_file(filename)
        text = util.cleanhtml(text)
        assert rev['pageid'] == rev_xl.iloc[index, 0]
        print("matched ",rev['revid'])
        print(rev['revid'],'==cleaned')
        # """
        #     Returns a tuple with:
        #      (number_chars, number_words, number_types, number_sentences, number_syllables, number_polysyllable_words,
        #         number_words_longer_4, number_words_longer_6, number_words_longer_10,
        #         number_words_longer_longer_13,
        #          , lix_index, )
        # """

        #  linsear_write_formula, , readability_consensus, infonoisescore,
        # logcontentlength, logreferences, logpagelinks, numimageslength, num_citetemplates, lognoncitetemplates,
        # num_categories, hasinfobox, lvl2headings, lvl3heading

        # calc = readcalc.ReadCalc(text)
        # t = calc.get_all_metrics()

        tool = language_check.LanguageTool('en-US')
        text = u'A sentence with a error in the Hitchhiker’s Guide tot he Galaxy'
        matches = tool.check(text)
        len(matches)

        tt = [round(util.check_grammar_error_rate_o(tool,text),2)]
        print(rev['revid'], tt)
        rev_xl.iloc[index, 34:35] = tt

        return rev_xl.iloc[index, :]


def clean():
    rev_train = pd.read_csv("readscore/all_score_train-c-output.csv",
                         dtype={0: 'int32', 1: 'int32', 2: 'object'})

    rev_test = pd.read_csv("readscore/all_score_test-c-output.csv",
                         dtype={0: 'int32', 1: 'int32', 2: 'object'})

    print(rev_train.shape)
    for index,row in rev_test.iterrows():
        rev_train.drop(rev_train[rev_train.revid ==row['revid']].index, inplace=True)

    print(rev_train.shape)

    rev_train.to_csv("readscore/all_score_train-c-output.csv")


def startCalcFeatures():
    # Load rules from incredibly high-tech datastore.

    # new_column = ['number_chars', 'number_words', 'number_types', 'number_sentences', 'number_syllables',
    #               'number_polysyllable_words',
    #               'difficult_words', 'number_words_longer_4', 'number_words_longer_6', 'number_words_longer_10',
    #               'number_words_longer_longer_13', 'flesch_reading_ease', 'flesch_kincaid_grade_level',
    #               'coleman_liau_index',
    #               'gunning_fog_index', 'smog_index', 'ari_index', 'lix_index', 'dale_chall_score','linsear_write_formula']
    global rev_xl
    new_column = ['grammar']

    rev_xl = rev_xl.reindex(columns=rev_xl.columns.tolist() + new_column)

    # calcFeatures([0,rev_xl.loc[rev_xl['article_revid'] == 491864508].squeeze()])

    # Perform classification.
    with multiprocessing.Pool() as p:
        result = p.map(calcFeatures, islice(rev_xl.iterrows(), 5))
        #result = p.map(calcFeatures, rev_xl.iterrows())

    final = pd.DataFrame(data=result)
    final.to_csv("readscore/all_score_train-c-output.csv")


if __name__ == "__main__":
    rev_xl = pd.read_csv("readscore/all_score_train-c-output.csv",
                         dtype={0: 'int32', 1: 'int32', 2: 'object'})
    startCalcFeatures()
