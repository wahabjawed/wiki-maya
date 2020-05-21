from itertools import cycle
from pprint import pprint

from numpy import interp
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, KFold
from sklearn.preprocessing import label_binarize, MinMaxScaler
from sklearn.linear_model import LinearRegression


class ExtraTreeT:

    def __init__(self, args):
        self.test_data_path = args[0]
        self.train_data_path = args[1]

        self.train = pd.read_csv(self.train_data_path)
        self.test = pd.read_csv(self.test_data_path)
        self.target_names = self.train['rating'].unique()

        self.features = ['infonoisescore', 'logcontentlength', 'logreferences', 'logpagelinks', 'numimageslength',
                         'num_citetemplates', 'lognoncitetemplates',
                         'num_categories', 'hasinfobox', 'lvl2headings', 'lvl3heading', 'number_chars', 'number_words',
                         'number_types', 'number_sentences', 'number_syllables',
                         'number_polysyllable_words', 'difficult_words', 'number_words_longer_4',
                         'number_words_longer_6', 'number_words_longer_10',
                         'number_words_longer_longer_13', 'flesch_reading_ease', 'flesch_kincaid_grade_level',
                         'coleman_liau_index',
                         'gunning_fog_index', 'smog_index', 'ari_index', 'lix_index',
                         'dale_chall_score', 'linsear_write_formula']

        cat = pd.Categorical(self.train['rating'], categories=['B', 'C', 'FA', 'GA', 'Start', 'Stub'], ordered=True)
        self.train_y, self.train_mapping = pd.factorize(cat)

        cat = pd.Categorical(self.test['rating'], categories=['B', 'C', 'FA', 'GA', 'Start', 'Stub'], ordered=True)
        self.test_y, self.test_mapping = pd.factorize(cat)
        # print('train: ', len(self.train))
        # print('test: ', len(self.test))

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(self.train[self.features])
        self.train_mm = scaler.transform(self.train[self.features])
        self.test_mm = scaler.transform(self.test[self.features])

    def hyperTune(self):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=10, stop=60, num=11)]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(5, 30, num=11)]
        # Minimum number of samples required to split a node
        min_samples_split = [1.0, 2, 3]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [0.5, 1, 2]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        pprint(random_grid)

        self.clf = ExtraTreesClassifier()
        rf_random = RandomizedSearchCV(estimator=self.clf, param_distributions=random_grid, n_iter=200, cv=5, verbose=2,
                                       random_state=42, n_jobs=-1)

        # Fit the random search model
        rf_random.fit(self.train[self.features], self.train_y)

        pprint(rf_random.best_params_)
        preds = self.target_names[rf_random.predict(self.test[self.features])]

        print(pd.crosstab(self.test['rating'], preds, rownames=['Actual Species'], colnames=['predicted']))
        print('Classification accuracy without selecting features: {:.3f}'
              .format(accuracy_score(self.test['rating'], preds)))
        return rf_random

    def learn(self):
        self.clf = ExtraTreesClassifier(n_estimators=300, max_depth=30)
        self.clf.fit(self.train[self.features], self.train_y)

        # kf = KFold(shuffle=True, n_splits=5)
        # scores = cross_val_score(self.clf, self.train[self.features], self.train_y, cv=kf, n_jobs=-1,
        #                          scoring='accuracy')
        # print(scores)
        # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        return self.clf

    def fetchScore(self):
        preds = self.target_names[self.clf.predict(self.test[self.features])]

        print(pd.crosstab(self.test['rating'], preds, rownames=['Actual Species'], colnames=['predicted']))
        print('Classification accuracy without selecting features: {:.3f}'
              .format(accuracy_score(self.test['rating'], preds)))

    def evaluate(self, model):
        predictions = self.target_names[model.predict(self.test[self.features])]
        print(pd.crosstab(self.test['rating'], predictions, rownames=['Actual Species'], colnames=['predicted']))
        print('Classification accuracy without selecting features: {:.3f}'
              .format(accuracy_score(self.test['rating'], predictions)))
