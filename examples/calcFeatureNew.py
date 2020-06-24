import os
import language_check
import pandas as pd
from readcalc import readcalc
from textstat.textstat import textstatistics

from maya.nltk import util


def calcFeatures(params):
    index, rev = params  # Multiprocessing...
    global rev_xl
    filename = "insert data path of the 2015 data from https://figshare.com/articles/English_Wikipedia_Quality_Asssessment_Dataset/1375406" + str(
        rev['revid'])
    if (os.path.exists(filename)):
        print(rev['revid'])
        text = util.read_file(filename)
        text = util.cleanhtml(text)
        text = text.replace('\'\'\'', '')
        assert rev['pageid'] == rev_xl.iloc[index, 0]
        print("matched ", rev['revid'])

        calc = readcalc.ReadCalc(text)
        textual_score = list(calc.get_all_metrics())

        text_stat = textstatistics()
        linsear_write_formula = round(text_stat.linsear_write_formula(text),2)
        textual_score.append(linsear_write_formula)

        grammar_score = len(tool.check(text))
        textual_score.append(grammar_score)

        rev_xl.iloc[index, 14:36] = textual_score

        print(rev_xl.iloc[index, :])

        if index % 10 == 0:
            rev_xl.to_csv(path)


def startCalcFeatures():
    # Load rules from incredibly high-tech datastore.

    new_column = ['number_chars', 'number_words', 'number_types', 'number_sentences', 'number_syllables',
                  'number_polysyllable_words',
                  'difficult_words', 'number_words_longer_4', 'number_words_longer_6', 'number_words_longer_10',
                  'number_words_longer_longer_13', 'flesch_reading_ease', 'flesch_kincaid_grade_level',
                  'coleman_liau_index',
                  'gunning_fog_index', 'smog_index', 'ari_index', 'lix_index', 'dale_chall_score',
                  'linsear_write_formula', 'grammar']
    global rev_xl

    rev_xl = rev_xl.reindex(columns=rev_xl.columns.tolist() + new_column)

    # Perform classification.
    for index, row in rev_xl.iterrows():
        calcFeatures([index, row])

    rev_xl.to_csv(path)


if __name__ == "__main__":
    path = "readscore/all_score_train-c-output.csv"

    rev_xl = pd.read_csv(path,
                         dtype={0: 'int32', 1: 'int32', 2: 'object'})
    tool = language_check.LanguageTool('en-US')
    startCalcFeatures()
