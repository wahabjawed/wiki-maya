from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

class RandomForest:

    def __init__(self, args):
        self.test_data_path = args[0]
        self.train_data_path = args[1]


        self.train = pd.read_csv(self.train_data_path)
        self.test = pd.read_csv(self.test_data_path)
        self.target_names = self.train['rating'].unique()


        print('train: ', len(self.train))
        print('test: ', len(self.test))


    def learn(self):
        self.features = self.train.columns[8:17]

        print(self.features)

        y = pd.factorize(self.train['rating'])[0]
        self.clf = RandomForestClassifier(n_jobs=5, random_state=0)
        self.clf.fit(self.train[self.features], y)


    def fetchScore(self):
        # self.clf.predict(self.test[self.features])
        # #
        # self.clf.predict_proba(self.test[self.features])[0:10]
        preds = self.target_names[self.clf.predict(self.test[self.features])]
        #
        print(preds[0:5])
        #
        print(pd.crosstab(self.test['rating'], preds, rownames=['Actual Species'], colnames=['predicted']))
        #

