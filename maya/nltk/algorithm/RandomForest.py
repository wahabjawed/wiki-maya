from itertools import cycle
from pprint import pprint

from numpy import interp
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, KFold
from sklearn.preprocessing import label_binarize, MinMaxScaler


class RandomForest:

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
                                                                                      'linsear_write_formula','grammar']


        self.train['score'] = self.train['rating'].apply(self.score_to_numeric)
        self.test['score'] = self.test['rating'].apply(self.score_to_numeric)
        self.classes = ['Stub', 'Start', 'C', 'B', 'GA', 'FA']

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(self.train[self.features])
        self.train_mm = scaler.transform(self.train[self.features])
        self.test_mm = scaler.transform(self.test[self.features])

    def evaluate(self, model):
        predictions = self.target_names[model.predict(self.test[self.features])]
        print(pd.crosstab(self.test['rating'], predictions, rownames=['Actual Species'], colnames=['predicted']))
        print('Classification accuracy without selecting features: {:.3f}'
              .format(accuracy_score(self.test['rating'], predictions)))

    def hyperTuneRandomForest(self):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=300, stop=800, num=25)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(20, 120, num=5)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 7]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        pprint(random_grid)

        self.clf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(estimator=self.clf, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                                       random_state=42, n_jobs=-1)

        # Fit the random search model
        rf_random.fit(self.train[self.features], self.train['score'])

        pprint(rf_random.best_params_)

        self.evaluate(rf_random)
        return rf_random

    def learn(self):
        self.clf = RandomForestClassifier(bootstrap=True, n_estimators=450, max_depth=55, n_jobs=5,
                                          random_state=42)

        self.clf.fit(self.train_mm, self.train['score'])

        # kf = KFold(shuffle=True, n_splits=5)
        # scores = cross_val_score(self.clf, self.train[self.features], self.train_y, cv=kf, n_jobs=-1,
        #                          scoring='accuracy')
        # print(scores)
        # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        return self.clf

    def featureImportance(self):
        scores2 = cross_val_score(self.clf, self.train[self.features],  self.train['score'], cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))

        ####finding important feature
        feature_imp = pd.Series(self.clf.feature_importances_, index=self.features).sort_values(ascending=False)

        #feature_imp
        sns.barplot(x=feature_imp, y=feature_imp.index)
        # Add labels to your graph
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Visualizing Important Features")
        plt.show()

    def fetchScore(self):

        preds = self.clf.predict(self.test_mm)
        preds = np.array([self.classes[x] for x in preds])

        print(pd.crosstab(self.test['rating'], preds, rownames=['Actual Species'], colnames=['predicted']))
        print('Classification accuracy without selecting features: {:.3f}'
              .format(accuracy_score(self.test['rating'], preds)))



