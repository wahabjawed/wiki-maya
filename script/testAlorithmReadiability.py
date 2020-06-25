import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from maya.nltk.algorithm.CARTT import CARTT
from maya.nltk.algorithm.ExtraTreeT import ExtraTreeT
from maya.nltk.algorithm.GradientBoosting import GBT
from maya.nltk.algorithm.KNNT import KNNT
from maya.nltk.algorithm.MLRT import MLRT
from maya.nltk.algorithm.MMNB import MMNB
from maya.nltk.algorithm.NNTR import NNTR
from maya.nltk.algorithm.RandomForest import RandomForest
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import f_classif

from maya.nltk.algorithm.SVMT import SVMT
from maya.nltk.algorithm.VCH import VCH


def executeRandomForest(train_data_path, test_data_path):
    random = RandomForest([test_data_path, train_data_path])
    random.learn()
    random.fetchScore()


def executeExtraTreeT(train_data_path, test_data_path):
    random = ExtraTreeT([test_data_path, train_data_path])
    random.learn()
    random.fetchScore()


def executeCART(train_data_path, test_data_path):
    random = CARTT([test_data_path, train_data_path])
    random.learn()
    random.fetchScore()


def executeSVM(train_data_path, test_data_path):
    random = SVMT([test_data_path, train_data_path])
    random.learn()
    random.fetchScore()


def executeKNN(train_data_path, test_data_path):
    random = KNNT([test_data_path, train_data_path])
    random.learn()
    random.fetchScore()


def executeMLRT(train_data_path, test_data_path):
    random = MLRT([test_data_path, train_data_path])
    random.learn()
    random.fetchScore()


def executeNNTR(train_data_path, test_data_path):
    random = NNTR([test_data_path, train_data_path])
    random.learn()
#   random.computeROC()


def tuneRandomForest(train_data_path, test_data_path):
    random = RandomForest([test_data_path, train_data_path])
    random.hyperTuneRandomForest()


def tuneExtraTreeT(train_data_path, test_data_path):
    random = ExtraTreeT([test_data_path, train_data_path])
    random.hyperTune()


def executeGBT(train_data_path, test_data_path):
    random = GBT([test_data_path, train_data_path])
    random.learn()
    random.fetchScore()


def executeMMNB(train_data_path, test_data_path):
    random = MMNB([test_data_path, train_data_path])
    random.learn()
    random.fetchScore()


def executeVCH(train_data_path, test_data_path):
    random = VCH([test_data_path, train_data_path])
    random.learn()


def tuneGBT(train_data_path, test_data_path):
    random = GBT([test_data_path, train_data_path])
    random.hyperTune()


def tuneCart(train_data_path, test_data_path):
    random = CARTT([test_data_path, train_data_path])
    random.hyperTune()


def tuneMLRT(train_data_path, test_data_path):
    random = MLRT([test_data_path, train_data_path])
    random.hyperTune()


def roc(train_data_path, test_data_path):
    random = RandomForest([test_data_path, train_data_path])
    random.computeROC()


def findBestFeatures(train_data_path):
    # ANOVA feature selection for numeric input and categorical output

    train = pd.read_csv(train_data_path)

    feature_st = ['infonoisescore', 'logcontentlength', 'logreferences', 'logpagelinks', 'numimageslength',
                  'num_citetemplates', 'lognoncitetemplates',
                  'num_categories', 'hasinfobox', 'lvl2headings', 'lvl3heading']

    feature_red = ['number_chars', 'number_words',
                   'number_types', 'number_sentences', 'number_syllables',
                   'number_polysyllable_words', 'difficult_words', 'number_words_longer_4', 'number_words_longer_6',
                   'number_words_longer_10',
                   'number_words_longer_longer_13', 'flesch_reading_ease', 'flesch_kincaid_grade_level',
                   'coleman_liau_index',
                   'gunning_fog_index', 'smog_index', 'ari_index', 'lix_index', 'dale_chall_score',
                   'linsear_write_formula', 'grammar']

    features = train[feature_red]

    y = pd.factorize(train['rating'])[0]
    # define feature selection
    fs = SelectKBest(score_func=f_classif, k=20)
    # apply feature selection
    fs.fit(features, y)

    ####finding important feature
    feature_imp = pd.Series(np.log(fs.scores_), index=feature_red).sort_values(ascending=False)

    # feature_imp
    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.show()


def varianceThreshold(train_data_path):
    train = pd.read_csv(train_data_path,low_memory=False)
    features = ['infonoisescore', 'logcontentlength', 'logreferences', 'logpagelinks', 'numimageslength',
                'num_citetemplates', 'lognoncitetemplates',
                'num_categories', 'hasinfobox', 'lvl2headings', 'lvl3heading', 'number_chars', 'number_words',
                'number_types', 'number_sentences', 'number_syllables',
                'number_polysyllable_words', 'difficult_words', 'number_words_longer_4',
                'number_words_longer_6', 'number_words_longer_10',
                'number_words_longer_longer_13', 'flesch_reading_ease', 'flesch_kincaid_grade_level',
                'coleman_liau_index',
                'gunning_fog_index', 'smog_index', 'ari_index', 'lix_index', 'dale_chall_score',
                'linsear_write_formula', 'grammar']

    y = pd.factorize(train['rating'])[0]
    v = VarianceThreshold()
    res = v.fit(train[features], y)
    print(res)

    ####finding important feature
    feature_imp = pd.Series(np.log(res.variances_), index=features).sort_values(ascending=False)

    # feature_imp
    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Variance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Features Variance")
    plt.show()


def tuneKNN(train_data_path, test_data_path):
    random = KNNT([test_data_path, train_data_path])
    random.hyperTune()


if __name__ == "__main__":
    train_data_path = "readscore/all_score_train-c-output.csv"
    test_data_path = "readscore/all_score_test-c-output.csv"
    findBestFeatures(train_data_path)
    executeRandomForest(train_data_path, test_data_path)
    # tuneRandomForest(train_data_path, test_data_path)
    # roc(train_data_path, test_data_path)
    executeExtraTreeT(train_data_path, test_data_path)
    # tuneExtraTreeT(train_data_path, test_data_path)
    executeCART(train_data_path, test_data_path)
    # tuneCart(train_data_path, test_data_path)
    executeSVM(train_data_path, test_data_path)
    executeMLRT(train_data_path, test_data_path)
    # tuneMLRT(train_data_path, test_data_path)
    executeKNN(train_data_path, test_data_path)
    # tuneKNN(train_data_path, test_data_path)
    executeGBT(train_data_path, test_data_path)
    # tuneGBT(train_data_path, test_data_path)
    executeVCH(train_data_path, test_data_path)
    executeNNTR(train_data_path, test_data_path)
    executeMMNB(train_data_path, test_data_path)
    varianceThreshold(train_data_path)
