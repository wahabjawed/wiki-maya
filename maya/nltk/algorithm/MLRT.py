from itertools import cycle
from pprint import pprint
from scipy.stats import randint

from numpy import interp
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, KFold, GridSearchCV
from sklearn.preprocessing import label_binarize, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class MLRT:

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
                         'gunning_fog_index', 'smog_index', 'ari_index', 'lix_index', 'dale_chall_score',
                         'linsear_write_formula']

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
        # Create the random grid
        # Setup the parameters and distributions to sample from: param_dist
        param_grid = {'multi_class': ['multinomial'],
                      'C': np.logspace(-2, 2, 10),
                      'solver': ['newton-cg']}

        pprint(param_grid)

        self.clf = LogisticRegression()
        rf_random = RandomizedSearchCV(estimator=self.clf, n_iter=200, param_distributions=param_grid, cv=5, verbose=2,
                                       n_jobs=-1)

        # Fit the random search model
        rf_random.fit(self.train[self.features], self.train_y)

        pprint(rf_random.best_params_)
        print("Best score is {}".format(rf_random.best_score_))

        preds = self.target_names[rf_random.predict(self.test[self.features])]

        print(pd.crosstab(self.test['rating'], preds, rownames=['Actual Species'], colnames=['predicted']))
        print('Classification accuracy without selecting features: {:.3f}'
              .format(accuracy_score(self.test['rating'], preds)))
        return rf_random

    def learn(self):
        self.clf = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=275)
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
        # print(classification_report(self.test['rating'], preds))

    def evaluate(self, model):
        predictions = self.target_names[model.predict(self.test[self.features])]
        print(pd.crosstab(self.test['rating'], predictions, rownames=['Actual Species'], colnames=['predicted']))
        print('Classification accuracy without selecting features: {:.3f}'
              .format(accuracy_score(self.test['rating'], predictions)))
