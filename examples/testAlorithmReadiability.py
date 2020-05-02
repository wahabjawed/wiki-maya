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



def findBestFeatures(train_data_path, test_data_path):
    # ANOVA feature selection for numeric input and categorical output

    train = pd.read_csv(train_data_path)
    feature_name = ['linsear_write_formula', 'dale_chall_readability_score', 'rix',
                    'spache_readability', 'dale_chall_readability_score_v2', 'reading_time', 'grammar',
                    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16","17", "18",
                    "19"]
    features = train[feature_name]

    target_names = train['rating'].unique()
    print(features)

    y = pd.factorize(train['rating'])[0]
    # define feature selection
    fs = SelectKBest(score_func=f_classif, k=17)
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
    train_data_path = "../data/2017_english_wikipedia_quality_dataset/datasets/training-set-n3.csv"
    test_data_path = "../data/2017_english_wikipedia_quality_dataset/datasets/test-set-n3.csv"
    #findBestFeatures(train_data_path, test_data_path)
    executeRandomForest(train_data_path, test_data_path)
    #tuneRandomForest(train_data_path, test_data_path)
