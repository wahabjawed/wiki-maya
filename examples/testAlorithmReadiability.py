import pandas as pd

from maya.nltk.algorithm.RandomForest import RandomForest
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


def executeRandomForest(train_data_path, test_data_path):
    random = RandomForest([test_data_path, train_data_path])
    random.learn()
    random.fetchScore()

def tuneRandomForest(train_data_path, test_data_path):
    random = RandomForest([test_data_path, train_data_path])
    random.hyperTuneRandomForest()

def roc(train_data_path, test_data_path):
    random = RandomForest([test_data_path, train_data_path])
    random.computeROC()

def findBestFeatures(train_data_path, test_data_path):
    # ANOVA feature selection for numeric input and categorical output

    train = pd.read_csv(train_data_path)
    # feature_name = ['linsear_write_formula', 'dale_chall_readability_score', 'rix',
    #                 'spache_readability', 'dale_chall_readability_score_v2', 'reading_time', 'grammar',
    #                 "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16","17", "18",
    #                 "19"]

    feature_name = ['infonoisescore', 'logcontentlength','logreferences','logpagelinks','numimageslength','num_citetemplates','lognoncitetemplates',
                         'num_categories','hasinfobox','lvl2headings','lvl3heading', 'number_chars', 'number_words', 'number_types', 'number_sentences', 'number_syllables',
     'number_polysyllable_words', 'difficult_words', 'number_words_longer_4', 'number_words_longer_6', 'number_words_longer_10',
     'number_words_longer_longer_13', 'flesch_reading_ease', 'flesch_kincaid_grade_level', 'coleman_liau_index',
     'gunning_fog_index', 'smog_index', 'ari_index', 'lix_index', 'dale_chall_score']

    features = train[feature_name]

    target_names = train['user_rating'].unique()
    print(features)

    y = pd.factorize(train['user_rating'])[0]
    # define feature selection
    fs = SelectKBest(score_func=f_classif, k=25)
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




if __name__ == "__main__":
    # train_data_path = "../data/2017_english_wikipedia_quality_dataset/datasets/training-set-n3.csv"
    # test_data_path = "../data/2017_english_wikipedia_quality_dataset/datasets/test-set-n3.csv"
    train_data_path = "readscore/all_score_train-c-output.csv"
    test_data_path = "readscore/all_score_test-c-output.csv"
    #findBestFeatures(train_data_path, test_data_path)
    #executeRandomForest(train_data_path, test_data_path)
    #tuneRandomForest(train_data_path, test_data_path)
    roc(train_data_path, test_data_path)
