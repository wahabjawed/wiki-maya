from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler


class GBT:

    def score_to_numeric(self, x):
        if x == 'Stub':
            return 0
        if x == 'Start':
            return 1
        if x == 'C':
            return 2
        if x == 'B':
            return 3
        if x == 'GA':
            return 4
        if x == 'FA':
            return 5

    def __init__(self, args):
        self.test_data_path = args[0]
        self.train_data_path = args[1]

        self.train = pd.read_csv(self.train_data_path, low_memory=False)
        self.test = pd.read_csv(self.test_data_path, low_memory=False)

        self.features = ['infonoisescore', 'logcontentlength', 'logreferences', 'logpagelinks', 'numimageslength',
                         'num_citetemplates', 'lognoncitetemplates',
                         'num_categories', 'lvl2headings', 'lvl3heading', 'number_chars', 'number_words',
                         'number_types', 'number_sentences', 'number_syllables',
                         'number_polysyllable_words', 'difficult_words', 'number_words_longer_4',
                         'number_words_longer_6', 'number_words_longer_10',
                         'number_words_longer_longer_13', 'flesch_reading_ease', 'flesch_kincaid_grade_level',
                         'coleman_liau_index',
                         'gunning_fog_index', 'smog_index', 'ari_index', 'lix_index',
                         'dale_chall_score', 'linsear_write_formula', 'grammar']

        self.train['score'] = self.train['rating'].apply(self.score_to_numeric)
        self.test['score'] = self.test['rating'].apply(self.score_to_numeric)
        self.classes = ['Stub', 'Start', 'C', 'B', 'GA', 'FA']

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(self.train[self.features])
        self.train_mm = scaler.transform(self.train[self.features])
        self.test_mm = scaler.transform(self.test[self.features])


    def hyperTune(self):
        # Create the random grid
        # Setup the parameters and distributions to sample from: param_dist
        param_grid = {
            'learning_rate': [0.15, 0.2, 0.1, 0.08, 0.07],
            'n_estimators': [125, 135, 145, 150, 165, 175, 185],
            'max_depth': [int(x) for x in np.linspace(5, 11, 6, endpoint=True)],
            'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2, 3, 4]}

        pprint(param_grid)

        self.clf = GradientBoostingClassifier()
        rf_random = RandomizedSearchCV(estimator=self.clf, n_iter=100, param_distributions=param_grid, cv=5, verbose=2,
                                       n_jobs=-1)

        # Fit the random search model
        rf_random.fit(self.train[self.features], self.train_y)

        pprint(rf_random.best_params_)
        print("Best score is {}".format(rf_random.best_score_))

        self.evaluate(rf_random)
        return rf_random

    def learn(self):
        self.clf = GradientBoostingClassifier(n_estimators=155, max_depth=7, min_samples_split=3, min_samples_leaf=1,
                                               learning_rate=0.07)
        #self.clf = GradientBoostingClassifier(n_estimators=100, max_depth=6)
        self.clf.fit(self.train[self.features], self.train['score'])
        # kf = KFold(shuffle=True, n_splits=5)
        # scores = cross_val_score(self.clf, self.train[self.features], self.train_y, cv=kf, n_jobs=-1,
        #                          scoring='accuracy')
        # print(scores)
        # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        return self.clf

    def fetchScore(self):
        preds = self.clf.predict(self.test[self.features])
        preds = np.array([self.classes[x] for x in preds])
        print(pd.crosstab(self.test['rating'], preds, rownames=['Actual Species'], colnames=['predicted']))
        print('Classification accuracy without selecting features: {:.3f}'
              .format(accuracy_score(self.test['rating'], preds)))

    def evaluate(self, model):
        predictions = model.predict(self.test[self.features])
        predictions = np.array([self.classes[x] for x in predictions])
        print(pd.crosstab(self.test['rating'], predictions, rownames=['Actual Species'], colnames=['predicted']))
        print('Classification accuracy without selecting features: {:.3f}'
              .format(accuracy_score(self.test['rating'], predictions)))
