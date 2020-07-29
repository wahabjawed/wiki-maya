import keras
import numpy as np
import pandas as pd
from keras import Sequential
from tensorflow.keras import initializers, optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU


class VCH:

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
                         'dale_chall_score', 'linsear_write_formula', 'grammar']

        self.train['score'] = self.train['rating'].apply(self.score_to_numeric)
        self.test['score'] = self.test['rating'].apply(self.score_to_numeric)
        self.classes = ['Stub', 'Start', 'C', 'B', 'GA', 'FA']

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(self.train[self.features])
        self.train_mm = scaler.transform(self.train[self.features])
        self.test_mm = scaler.transform(self.test[self.features])

    # define baseline model
    def baseline_model(self):
        initializer = initializers.he_uniform()
        self.clf = Sequential()
        self.clf.add(Dense(128, input_dim=32, kernel_initializer=initializer))
        self.clf.add(LeakyReLU())
        self.clf.add(Dense(512, kernel_initializer=initializer))
        self.clf.add(LeakyReLU())
        self.clf.add(Dropout(0.1))
        self.clf.add(Dense(256, kernel_initializer=initializer))
        self.clf.add(LeakyReLU())
        self.clf.add(Dropout(0.1))
        self.clf.add(Dense(6, activation='softmax'))  # Final Layer using Softmax
        optimizer = optimizers.Adamax(0.0008)
        self.clf.compile(loss='sparse_categorical_crossentropy',
                         optimizer=optimizer, metrics=['accuracy'])
        return self.clf

    def learn(self):
        # group / ensemble of models
        estimator = []
        estimator.append(('KNN',KNeighborsClassifier(n_neighbors=12,p=6)))
        estimator.append(('LR',
                          LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter=275)))
        estimator.append(('RFC', RandomForestClassifier(bootstrap=True, n_estimators=450, max_depth=65, n_jobs=5,
                                          min_samples_leaf=1, min_samples_split=5, random_state=42)))
        estimator.append(('GB',  GradientBoostingClassifier(n_estimators=145, max_depth=6, min_samples_split=3, min_samples_leaf=1,
                                              learning_rate=0.07)))
        estimator.append(('NN',  KerasClassifier(build_fn=self.baseline_model, epochs=35, verbose=2)))

        # Voting Classifier with hard voting
        self.clf = VotingClassifier(estimators=estimator, voting='hard')
        self.clf.fit(self.train[self.features], self.train['score'])
        y_pred = self.clf.predict(self.test[self.features])
        y_pred = np.array([self.classes[x] for x in y_pred])
        print(y_pred)

        # using accuracy_score metric to predict accuracy
        print(pd.crosstab(self.test['rating'], y_pred, rownames=['Actual Species'], colnames=['predicted']))
        print('Classification accuracy hard: {:.3f}'
              .format(accuracy_score(self.test['rating'], y_pred)))

        # Voting Classifier with soft voting
        self.clf = VotingClassifier(estimators=estimator, voting='soft')
        self.clf.fit(self.train_mm, self.train['score'])
        y_pred = self.clf.predict(self.test_mm)
        y_pred = np.array([self.classes[x] for x in y_pred])
        print(y_pred)
        # using accuracy_score metric to predict accuracy
        print(pd.crosstab(self.test['rating'], y_pred, rownames=['Actual Species'], colnames=['predicted']))
        print('Classification accuracy soft: {:.3f}'
              .format(accuracy_score(self.test['rating'], y_pred)))
        return self.clf


    def fetchScore(self):
        preds = self.clf.predict(self.test[self.features])
        preds = np.array([self.classes[x] for x in preds])
        print(pd.crosstab(self.test['rating'], preds, rownames=['Actual Species'], colnames=['predicted']))
        print('Classification accuracy without selecting features: {:.3f}'
              .format(accuracy_score(self.test['rating'], preds)))



