import pandas as pd

from maya.nltk.algorithm.CARTT import CARTT
from maya.nltk.algorithm.ExtraTreeT import ExtraTreeT
from maya.nltk.algorithm.GradientBoosting import GBT
from maya.nltk.algorithm.KNNT import KNNT
from maya.nltk.algorithm.MLRT import MLRT
from maya.nltk.algorithm.MMNB import MMNB
from maya.nltk.algorithm.NNT import NNT
from maya.nltk.algorithm.RandomForest import RandomForest
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
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

def executeNNT(train_data_path, test_data_path):
    random = NNT([test_data_path, train_data_path])
    random.learn()

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

def findBestFeatures(train_data_path, test_data_path):
    # ANOVA feature selection for numeric input and categorical output

    train = pd.read_csv(train_data_path)


    feature_name = ['infonoisescore', 'logcontentlength','logreferences','logpagelinks','numimageslength','num_citetemplates','lognoncitetemplates',
                         'num_categories','hasinfobox','lvl2headings','lvl3heading', 'number_chars', 'number_words', 'number_types', 'number_sentences', 'number_syllables',
     'number_polysyllable_words', 'difficult_words', 'number_words_longer_4', 'number_words_longer_6', 'number_words_longer_10',
     'number_words_longer_longer_13', 'flesch_reading_ease', 'flesch_kincaid_grade_level', 'coleman_liau_index',
     'gunning_fog_index', 'smog_index', 'ari_index', 'lix_index', 'dale_chall_score']

    features = train[feature_name]

    target_names = train['rating'].unique()
    print(features)

    y = pd.factorize(train['rating'])[0]
    # define feature selection
    fs = SelectKBest(score_func=f_classif, k=28)
    # apply feature selection
    X_selected = fs.fit_transform(features, y)
    print(X_selected.shape)

    mask = fs.get_support()  # list of booleans
    new_features = []  # The list of your K best features

    for bool, feature in zip(mask, feature_name):
        if bool:
            new_features.append(feature)
    print(new_features)
    # dataframe = pd.DataFrame(fit_transofrmed_features, columns=new_features)


def tuneKNN(train_data_path, test_data_path):
    random = KNNT([test_data_path, train_data_path])
    random.hyperTune()


if __name__ == "__main__":
    # train_data_path = "../data/2017_english_wikipedia_quality_dataset/datasets/training-set-n3.csv"
    # test_data_path = "../data/2017_english_wikipedia_quality_dataset/datasets/test-set-n3.csv"
    train_data_path = "readscore/all_score_train-c-output.csv"
    test_data_path = "readscore/all_score_test-c-output.csv"
    #findBestFeatures(train_data_path, test_data_path)
    executeRandomForest(train_data_path, test_data_path)
    #tuneRandomForest(train_data_path, test_data_path)
    #roc(train_data_path, test_data_path)
    #executeExtraTreeT(train_data_path, test_data_path)
    #tuneExtraTreeT(train_data_path, test_data_path)
    #executeCART(train_data_path, test_data_path)
    #tuneCart(train_data_path, test_data_path)
    #executeSVM(train_data_path, test_data_path)
    #executeMLRT(train_data_path, test_data_path)
    #tuneMLRT(train_data_path, test_data_path)
    #executeKNN(train_data_path, test_data_path)
    #tuneKNN(train_data_path, test_data_path)
    #executeGBT(train_data_path, test_data_path)
    #tuneGBT(train_data_path, test_data_path)
    #executeVCH(train_data_path, test_data_path)
    #executeNNT(train_data_path, test_data_path)
    #executeMMNB(train_data_path, test_data_path)
