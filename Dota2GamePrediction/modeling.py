import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split


class RunningModel():
    def __init__(self, transformed_df, model, target_field, scaling=True):
        self.transformed_df = transformed_df.copy()
        self.model = model
        self.target_field = target_field
        self.scaling = scaling
        self.Features = self.get_features()
        self.Target = self.get_target()

    def get_target(self):
        return self.transformed_df[self.target_field].values

    def get_features(self):
        return self.transformed_df.drop([self.target_field], axis=1)

    def running_model(self, test_size=0.3):
        X_train_unscaled, X_test_unscaled, self.y_train, self.y_test = \
            train_test_split(self.Features, self.Target, test_size=test_size, random_state=1)
        if self.scaling:
            self.scaler = StandardScaler().fit(X_train_unscaled)
            self.X_train = self.scaler.transform(X_train_unscaled)
            self.X_test = self.scaler.transform(X_test_unscaled)
        else:
            self.X_train = X_train_unscaled
            self.X_test = X_test_unscaled
        self.model.fit(self.X_train, self.y_train)

    def model_estimator(self):
        for X, y, label in [[self.X_train, self.y_train, 'Train'],
                            [self.X_test, self.y_test, 'Test']]:
            y_pred = self.model.predict(X)
            cnf_matrix = metrics.confusion_matrix(y, y_pred) #.astype(float)
            self.plot_confusion_matrix(cm=cnf_matrix,
                                       classes=[1,2],
                                       title='{} dataset'.format(label))
            self.score(cnf_matrix)

    def score(self, cm):
        correct = cm[0][0] + cm[1][1]
        total = cm.sum()
        wrong = total - correct
        score = 100 * (float(correct - wrong) / total)
        print 'score = {0:.2f}'.format(score)

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        # print(cm)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
