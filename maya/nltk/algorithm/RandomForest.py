from pprint import pprint

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, KFold
from sklearn.preprocessing import label_binarize


class RandomForest:

    def __init__(self, args):
        self.test_data_path = args[0]
        self.train_data_path = args[1]

        self.train = pd.read_csv(self.train_data_path)
        self.test = pd.read_csv(self.test_data_path)
        self.target_names = self.train['rating'].unique()

        # self.features = ['linsear_write_formula', 'dale_chall_readability_score', 'rix',
        #                  'spache_readability', 'dale_chall_readability_score_v2', 'reading_time', 'grammar',
        #                  "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "17", "18",
        #                  "19"]

        self.features = ['infonoisescore', 'logcontentlength', 'logreferences', 'logpagelinks', 'numimageslength',
                         'num_citetemplates', 'lognoncitetemplates',
                         'num_categories', 'hasinfobox', 'lvl2headings', 'lvl3heading', 'number_chars', 'number_words',
                         'number_types', 'number_sentences', 'number_syllables',
                         'number_polysyllable_words', 'difficult_words', 'number_words_longer_4',
                         'number_words_longer_6', 'number_words_longer_10',
                         'number_words_longer_longer_13', 'flesch_reading_ease', 'flesch_kincaid_grade_level',
                         'coleman_liau_index',
                         'gunning_fog_index', 'smog_index', 'ari_index', 'lix_index', 'dale_chall_score']

        # self.train_y,self.train_mapping = pd.factorize(self.train['rating'])[0]
        # self.test_y,self.test_mapping = pd.factorize(self.test['rating'])[0]

        cat = pd.Categorical(self.train['rating'], categories=['B', 'C', 'FA', 'GA', 'Start', 'Stub'])
        self.train_y, self.train_mapping = pd.factorize(cat)

        cat = pd.Categorical(self.train['rating'], categories=['B', 'C', 'FA', 'GA', 'Start', 'Stub'])
        self.test_y, self.test_mapping = pd.factorize(cat)
        # print('train: ', len(self.train))
        # print('test: ', len(self.test))

    def hyperTuneRandomForest(self):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
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
        rf_random = RandomizedSearchCV(estimator=self.clf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                       random_state=42, n_jobs=-1)

        # Fit the random search model
        rf_random.fit(self.train[self.features], self.train_y)

        pprint(rf_random.best_params_)
        return rf_random

    def learn(self):
        # self.features = self.train.columns[16:38]

        # print(self.features)

        # from stats
        # self.features_imp = ['linsear_write_formula', 'reading_time', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '14', '16', '19']

        # from random forest
        self.features_imp = ['infonoisescore', 'logcontentlength', 'logreferences', 'logpagelinks', 'numimageslength',
                             'num_citetemplates', 'lognoncitetemplates',
                             'num_categories', 'lvl2headings', 'lvl3heading', 'number_chars', 'number_words',
                             'number_types', 'number_sentences', 'number_syllables',
                             'number_polysyllable_words', 'difficult_words', 'number_words_longer_4',
                             'number_words_longer_6', 'number_words_longer_10',
                             'number_words_longer_longer_13', 'flesch_reading_ease', 'flesch_kincaid_grade_level',
                             'coleman_liau_index',
                             'gunning_fog_index', 'smog_index', 'ari_index', 'lix_index', 'dale_chall_score']

        self.clf = RandomForestClassifier(bootstrap=True, n_estimators=400, max_depth=100, n_jobs=5,
                                          max_features='sqrt',
                                          min_samples_leaf=1, min_samples_split=5, random_state=42)
        self.clf.fit(self.train[self.features], self.train_y)

        kf = KFold(shuffle=True, n_splits=5)
        scores = cross_val_score(self.clf, self.train[self.features], self.train_y, cv=kf, n_jobs=-1,
                                 scoring='accuracy')
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        # self.clf2 = RandomForestClassifier(bootstrap=True,n_estimators=400, max_depth=100, n_jobs=5, max_features='sqrt',
        #                                   min_samples_leaf=1,min_samples_split=5,random_state=42)
        # self.clf2.fit(self.train[self.features_imp], self.y)
        #
        # scores2 = cross_val_score(self.clf2, self.train[self.features_imp], self.y, cv=5)
        # print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))
        #
        # ####finding important feature
        # feature_imp = pd.Series(self.clf.feature_importances_, index=self.features).sort_values(ascending=False)
        #
        #
        # #feature_imp
        # sns.barplot(x=feature_imp, y=feature_imp.index)
        # # Add labels to your graph
        # plt.xlabel('Feature Importance Score')
        # plt.ylabel('Features')
        # plt.title("Visualizing Important Features")
        # plt.show()
        return self.clf

    def computeROC(self):
        # Binarize the output
        self.by = label_binarize(self.train_y, classes=[0, 1, 2, 4, 5])
        self.byy = label_binarize(self.test_y, classes=[0, 1, 2, 4, 5])
        self.n_classes = self.by.shape[1]

        self.clf = RandomForestClassifier(bootstrap=True, n_estimators=400, max_depth=100, n_jobs=5,
                                          max_features='sqrt',
                                          min_samples_leaf=1, min_samples_split=5, random_state=42)

        self.y_score = self.clf.fit(self.train[self.features], self.by).decision_function(self.test[self.features])

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.byy[:, i], self.y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(self.byy.ravel(), self.y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        lw = 2
        plt.plot(fpr[2], tpr[2], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    def fetchScore(self):
        # self.clf.predict(self.test[self.features])
        # #
        # self.clf.predict_proba(self.test[self.features])[0:10]
        preds = self.target_names[self.clf.predict(self.test[self.features])]
        # preds2 = self.target_names[self.clf2.predict(self.test[self.features_imp])]
        #
        # print(preds[0:5])
        #
        print(pd.crosstab(self.test['rating'], preds, rownames=['Actual Species'], colnames=['predicted']))
        print('Classification accuracy without selecting features: {:.3f}'
              .format(accuracy_score(self.test['rating'], preds)))
        # print(pd.crosstab(self.test['rating'], preds2, rownames=['Actual Species'], colnames=['predicted']))
        # print('Classification accuracy with selecting features: {:.3f}'
        #       .format(accuracy_score(self.test['rating'], preds2)))

    def evaluate(self, model):
        predictions = self.target_names[model.predict(self.test[self.features])]
        print(pd.crosstab(self.test['rating'], predictions, rownames=['Actual Species'], colnames=['predicted']))
        print('Classification accuracy without selecting features: {:.3f}'
              .format(accuracy_score(self.test['rating'], predictions)))
