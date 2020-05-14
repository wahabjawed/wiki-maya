import glob
import multiprocessing
import os
from itertools import islice

import language_check
import pandas as pd
from maya.nltk import util
from maya.nltk.textstats.textstats import textstatistics
from readcalc import readcalc


def calcFeatures(params):
    index, rev = params  # Multiprocessing...
    filename = "/Users/abdulwahab/Desktop/internship/dataset/2015_english_wikipedia_quality_dataset/revisiondata/" + str(
        rev['revid'])
    if (os.path.exists(filename)):
        print(rev['revid'])
        text = util.read_file(filename)
        text = util.cleanhtml(text)
        assert rev['pageid'] == rev_xl.iloc[index, 0]

        # """
        #     Returns a tuple with:
        #      (number_chars, number_words, number_types, number_sentences, number_syllables, number_polysyllable_words,
        #         difficult_words, number_words_longer_4, number_words_longer_6, number_words_longer_10,
        #         number_words_longer_longer_13, flesch_reading_ease, flesch_kincaid_grade_level, coleman_liau_index,
        #         gunning_fog_index, smog_index, ari_index, lix_index, dale_chall_score)
        # """

        # pageid, revid, user_rating,
        # difficult_words, linsear_write_formula, readability_consensus, infonoisescore, logcontentlength, \
        # logreferences, logpagelinks, numimageslength, num_citetemplates, lognoncitetemplates, num_categories,\
        # hasinfobox, lvl2headings, lvl3heading

        #flesch_reading_ease, flesch_kincaid_grade, \
        # smog_index, coleman_liau_index, automated_readability_index, dale_chall_readability_score,gunning_fog_index




        calc = readcalc.ReadCalc(text)
        t = calc.get_all_metrics()
        tt = [round(var,2) for var in t]
        rev_xl.iloc[index, 14:35] = tt
        # stat = textstatistics_v2(ctext)
        # stat_val = stat.compute()
        # rev_xl.iloc[index, 18:22] = stat_val

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


def calcFeatures():
    # Load rules from incredibly high-tech datastore.

    new_column = ['number_chars', 'number_words', 'number_types', 'number_sentences', 'number_syllables',
                  'number_polysyllable_words',
                  'difficult_words', 'number_words_longer_4', 'number_words_longer_6', 'number_words_longer_10',
                  'number_words_longer_longer_13', 'flesch_reading_ease', 'flesch_kincaid_grade_level',
                  'coleman_liau_index',
                  'gunning_fog_index', 'smog_index', 'ari_index', 'lix_index', 'dale_chall_score']

    fileDir = os.path.dirname(os.path.realpath('__file__'))

    rev_xl = pd.read_csv("readscore/all_score_train-c.csv",
                         dtype={0: 'int32', 1: 'int32', 2: 'object'})

    rev_xl = rev_xl.reindex(columns=rev_xl.columns.tolist() + new_column)

    # calcFeatures([0,rev_xl.loc[rev_xl['article_revid'] == 491864508].squeeze()])

    # Perform classification.
    with multiprocessing.Pool() as p:
        # result = p.map(calcFeatures, islice(rev_xl.iterrows(), 5))
        result = p.map(calcFeatures, rev_xl.iterrows())

    final = pd.DataFrame(data=result)
    final.to_csv("readscore/all_score_train-c-output.csv")


if __name__ == "__main__":
    rev_xl = pd.read_csv("readscore/all_score_train-c.csv",
                         dtype={0: 'int32', 1: 'int32', 2: 'object'})
    clean()
