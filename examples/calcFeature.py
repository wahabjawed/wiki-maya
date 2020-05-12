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
    filename = os.path.join(fileDir,
                            "../data/2017_english_wikipedia_quality_dataset/revisiondata/" + str(rev['article_revid']))
    if (os.path.exists(filename)):
        print(rev['article_revid'])
        text = util.read_file(filename)
        ctext = util.cleanhtml(text)
        stat = textstatistics(ctext)
        stat_val = stat.compute()
        assert rev['article_pageid'] == rev_xl.iloc[index, 0]
        rev_xl.iloc[index, 5:17] = stat_val
        rev_xl.iloc[index, 17] = util.check_grammar_error_rate(tool, ctext, stat.v_sentence_count)

        calc = readcalc.ReadCalc(text)
        t = calc.get_all_metrics()
        tt = [round(var,2) for var in t]
        rev_xl.iloc[index, 18:37] = tt
        # stat = textstatistics_v2(ctext)
        # stat_val = stat.compute()
        # rev_xl.iloc[index, 18:22] = stat_val

        return rev_xl.iloc[index, :]



if __name__ == "__main__":
    # Load rules from incredibly high-tech datastore.
    rev_text = []
    tool = language_check.LanguageTool('en-US')
    new_column = ['flesch_kincaid_grade', 'smog_index', 'coleman_liau_index',
                  'automated_readability_index', 'linsear_write_formula', 'dale_chall_readability_score', 'gunning_fog',
                  'lix', 'rix',
                  'spache_readability', 'dale_chall_readability_score_v2', 'reading_time', 'grammar',
                  "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19"]


    fileDir = os.path.dirname(os.path.realpath('__file__'))
    rev_files = glob.glob("../data/2017_english_wikipedia_quality_dataset/revisiondata/*")
    for file in rev_files:
        file_name = os.path.basename(file)
        rev_text.append((file_name, file))

    rev_xl = pd.read_csv("../data/2017_english_wikipedia_quality_dataset/datasets/training-set.tsv", delimiter="\t",
                         dtype={0: 'int32', 1: 'int32', 2: 'int32', 3: 'int32', 4: 'object'})

    rev_xl = rev_xl.reindex(columns=rev_xl.columns.tolist() + new_column)

    # calcFeatures([0,rev_xl.loc[rev_xl['article_revid'] == 491864508].squeeze()])

    # Perform classification.
    with multiprocessing.Pool() as p:
        # result = p.map(calcFeatures, islice(rev_xl.iterrows(), 5))
        result = p.map(calcFeatures, rev_xl.iterrows())

    # rev_xl.to_csv("../data/2017_english_wikipedia_quality_dataset/datasets/test-set-n.csv")
    final = pd.DataFrame(data=result)
    final.to_csv("../data/2017_english_wikipedia_quality_dataset/datasets/training-set-n3.csv")
