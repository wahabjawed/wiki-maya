from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from numpy import interp
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, label_binarize
from tensorflow.keras import initializers, optimizers
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras import backend as K

class NNTR:

    def __init__(self, args):
        self.data_path = "readscore/all_score.csv"

        self.train = pd.read_csv(self.data_path, low_memory=False)

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

        self.classes = ['Stub', 'Start', 'C', 'B', 'GA', 'FA']

        self.by = label_binarize(self.train['rating'], classes=self.classes)

        scaler = MinMaxScaler(feature_range=(-1, 1))

        scaler.fit(self.train[self.features])
        self.train_mm = scaler.transform(self.train[self.features])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.train_mm,
            self.train['rating'], test_size=0.10, random_state=2)

    def auroc(self, y_true, y_pred):
        return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

    def auc(self, y_true, y_pred):
        auc = tf.metrics.AUC(y_true, y_pred)[1]
        K.get_session().run(tf.local_variables_initializer())
        return auc

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

        self.estimator = KerasClassifier(build_fn=self.baseline_model, epochs=1, verbose=1)

        self.estimator.fit(self.X_train, self.y_train)
        self.score = self.estimator.predict(self.X_test)
        #roc = roc_auc_score(self.y_test, self.score)
        acc = self.estimator.score(self.X_test, self.y_test)
        print(pd.crosstab(self.y_test, self.score, rownames=['Actual Species'], colnames=['predicted']))
        print(acc)
        #print(roc)


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
