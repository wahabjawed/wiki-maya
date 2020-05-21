from itertools import cycle
from pprint import pprint

import keras
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from scipy.stats import randint

from numpy import interp
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, classification_report
from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, label_binarize


class NNT:

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
                         'gunning_fog_index', 'smog_index', 'ari_index', 'lix_index', 'dale_chall_score','linsear_write_formula']

        cat = pd.Categorical(self.train['rating'], categories=['B', 'C', 'FA', 'GA', 'Start', 'Stub'], ordered=True)
        self.train_y, self.train_mapping = pd.factorize(cat)

        cat = pd.Categorical(self.test['rating'], categories=['B', 'C', 'FA', 'GA', 'Start', 'Stub'], ordered=True)
        self.test_y, self.test_mapping = pd.factorize(cat)
        # print('train: ', len(self.train))
        # print('test: ', len(self.test))


        self.classes = ['B', 'C', 'FA', 'GA', 'Start', 'Stub']
        self.by = label_binarize(self.train['rating'], classes=self.classes)
        self.byy = label_binarize(self.test['rating'], classes=self.classes)
        print(np.array(self.train[self.features]).shape)

        scaler = MinMaxScaler(feature_range=(0, 1))

        scaler.fit(self.train[self.features])
        self.train_mm = scaler.transform(self.train[self.features])
        self.test_mm = scaler.transform(self.test[self.features])

    # define baseline model
    def baseline_model(self):
        self.clf = Sequential()
        self.clf.add(Dense(256,input_dim=31, activation='relu'))
        self.clf.add(Dense(512, activation='relu'))
        self.clf.add(Dense(6, activation='softmax'))  # Final Layer using Softmax

        self.clf.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
        return self.clf

    def learn(self):
        estimator = KerasClassifier(build_fn=self.baseline_model, epochs=30, verbose=2)
        kfold = KFold(n_splits=5, shuffle=True)
        results = cross_val_score(estimator, self.train_mm, self.by,
                                  cv=kfold, error_score='raise')
        print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

        estimator.fit(self.train_mm, self.by)
        score = estimator.predict(self.test_mm)
        scores = np.array([self.classes[x] for x in score])
        acc = estimator.score(self.test_mm, self.byy)
        print(pd.crosstab(self.test['rating'], scores, rownames=['Actual Species'], colnames=['predicted']))
        print(acc)

    def tune(self):
        # grid search epochs, batch size and optimizer

        # param_grid = {
        #     'tfidf__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3)],
        #     'tfidf__use_idf': [True, False],
        #     'kc__epochs': [10, 100, ],
        #     'kc__dense_nparams': [32, 256, 512],
        #     'kc__init': ['uniform', 'zeros', 'normal', ],
        #     'kc__batch_size': [2, 16, 32],
        #     'kc__optimizer': ['RMSprop', 'Adam', 'Adamax', 'sgd'],
        #     'kc__dropout': [0.5, 0.4, 0.3, 0.2, 0.1, 0]
        # }

        optimizers = ['rmsprop', 'adam']
        init = ['glorot_uniform', 'normal', 'uniform']
        epochs = [10, 20, 30]
        batches = [5, 10, 20]
        param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
        grid = GridSearchCV(estimator=self.baseline_model, param_grid=param_grid)
        grid_result = grid.fit(self.train_mm, self.by)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

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
