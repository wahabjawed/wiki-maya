import keras
import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
from tensorflow.keras import initializers

class RocCallback(Callback):

    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_train = self.model.predict_proba(self.x)
        roc_train = roc_auc_score(self.y, y_pred_train)
        y_pred_val = self.model.predict_proba(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train, 4)), str(round(roc_val, 4))),
              end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


class NNT:

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
        self.test = pd.read_csv(self.test_data_path)
        self.target_names = self.train['rating'].unique()

        self.features = ['infonoisescore', 'logcontentlength', 'logreferences', 'logpagelinks', 'numimageslength',
                         'num_citetemplates', 'lognoncitetemplates',
                         'num_categories', 'lvl2headings', 'lvl3heading', 'number_chars', 'number_words',
                         'number_types', 'number_sentences', 'number_syllables',
                         'number_polysyllable_words', 'difficult_words', 'number_words_longer_4',
                         'number_words_longer_6', 'number_words_longer_10',
                         'number_words_longer_longer_13', 'flesch_reading_ease', 'flesch_kincaid_grade_level',
                         'coleman_liau_index',
                         'gunning_fog_index', 'smog_index', 'ari_index', 'lix_index', 'dale_chall_score',
                         'linsear_write_formula', 'grammar']

        # cat = pd.Categorical(self.train['rating'], categories=['B', 'C', 'FA', 'GA', 'Start', 'Stub'], ordered=True)
        # self.train_y, self.train_mapping = pd.factorize(cat)
        #
        # cat = pd.Categorical(self.test['rating'], categories=['B', 'C', 'FA', 'GA', 'Start', 'Stub'], ordered=True)
        # self.test_y, self.test_mapping = pd.factorize(cat)

        self.train['score'] = self.train['rating'].apply(self.score_to_numeric)
        self.test['score'] = self.test['rating'].apply(self.score_to_numeric)
        # print('train: ', len(self.train))
        # print('test: ', len(self.test))

        self.classes = ['Stub', 'Start', 'C', 'B', 'GA', 'FA']
        # self.by = label_binarize(self.train['rating'], classes=self.classes)
        # self.byy = label_binarize(self.test['rating'], classes=self.classes)
        # print(np.array(self.train[self.features]).shape)

        scaler = MinMaxScaler(feature_range=(0, 1))

        scaler.fit(self.train[self.features])
        self.train_mm = scaler.transform(self.train[self.features])
        self.test_mm = scaler.transform(self.test[self.features])

    # define baseline model
    def baseline_model(self):
        initializer = initializers.he_uniform()
        self.clf = Sequential()
        self.clf.add(Dense(160, input_dim=31, activation='tanh',
                           bias_initializer='uniform'))
        self.clf.add(Dropout(0.3))
        self.clf.add(Dense(1024, activation='relu', bias_initializer='uniform'))
        self.clf.add(Dropout(0.50))
        self.clf.add(Dense(256, activation='relu'))
        self.clf.add(Dense(6, activation='softmax'))  # Final Layer using Softmax
        optimizer = keras.optimizers.Adamax(lr=0.0017)
        self.clf.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.clf

    def learn(self):

       # K.tensorflow_backend._get_available_gpus()

        estimator = KerasClassifier(build_fn=self.baseline_model, epochs=70, verbose=2)
        # kfold = KFold(n_splits=5, shuffle=True)
        # results = cross_val_score(estimator, self.train_mm, self.train_y,
        #                           cv=kfold, error_score='raise')
        # print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

        # roc = RocCallback(training_data=(X_train, y_train),
        #                   validation_data=(X_test, y_test))

        estimator.fit(self.train_mm, self.train['score'])
        score = estimator.predict(self.test_mm)
        scores = np.array([self.classes[x] for x in score])
        acc = estimator.score(self.test_mm, self.test['score'])
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
