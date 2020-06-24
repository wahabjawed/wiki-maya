from itertools import cycle

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import interp
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Dropout,LeakyReLU
from tensorflow.keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, auc, roc_curve
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, label_binarize
from tensorflow.keras import layers
from tensorflow.keras import initializers, optimizers
import matplotlib.pyplot as plt

class NNTR:

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

        self.features = ['infonoisescore', 'logcontentlength', 'hasinfobox', 'logreferences', 'logpagelinks', 'numimageslength',
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
        self.classes = ['Stub', 'Start', 'C', 'B', 'GA', 'FA']
        self.classes_n = [0, 1, 2, 3, 4, 5]

        self.by = label_binarize(self.train['rating'], classes=self.classes)
        self.byy = label_binarize(self.test['rating'], classes=self.classes)

        # print('train: ', len(self.train))
        # print('test: ', len(self.test))


        # self.by = label_binarize(self.train['rating'], classes=self.classes)
        # self.byy = label_binarize(self.test['rating'], classes=self.classes)
        # print(np.array(self.train[self.features]).shape)

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
        self.clf.add(Dropout(0.1))
        self.clf.add(Dense(512, kernel_initializer=initializer))
        self.clf.add(LeakyReLU())
        self.clf.add(Dropout(0.1))
        self.clf.add(Dense(256, kernel_initializer=initializer))
        self.clf.add(LeakyReLU())
        self.clf.add(Dropout(0.1))
        self.clf.add(Dense(6, activation='softmax'))  # Final Layer using Softmax
        optimizer = optimizers.Adamax()
        self.clf.compile(loss='categorical_crossentropy',
                         optimizer=optimizer, metrics=['accuracy'])
        return self.clf

    def learn(self):

        self.estimator = KerasClassifier(build_fn=self.baseline_model, epochs=65, verbose=1)
        # kfold = KFold(n_splits=5, shuffle=True)
        # results = cross_val_score(estimator, self.train_mm, self.train_y,
        #                           cv=kfold, error_score='raise')
        # print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

        # roc = RocCallback(training_data=(X_train, y_train),
        #                   validation_data=(X_test, y_test))

        self.estimator.fit(self.train_mm, self.by)
        self.score = self.estimator.predict(self.test_mm)
        scores = np.array([self.classes[x] for x in self.score])
        acc = self.estimator.score(self.test_mm, self.byy)
        print(pd.crosstab(self.test['rating'], scores, rownames=['Actual Species'], colnames=['predicted']))
        print(acc)


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

    def computeROC(self):
        # Binarize the output


        self.n_classes = self.by.shape[1]
        self.y_score = label_binarize(self.score, classes=self.classes_n)
        kf = KFold(shuffle=True, n_splits=5)
        # roc_auc_score = cross_val_score(self.estimator, self.train_mm, self.by, cv=kf, scoring='roc_auc')
        # print('roc_auc_score ', np.mean(roc_auc_score), roc_auc_score)


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

        lw = 2
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= self.n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue','yellow','pink'])
        for i, color in zip(range(self.n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(self.classes[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()
