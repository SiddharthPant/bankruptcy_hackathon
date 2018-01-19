import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class Model(object):
    """
    Encapsulates model and their related train test and fit methods
    """

    def __init__(self, x_train, y_train, x_test, y_test, model):
        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test
        self.model = model
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = model.predict(x_test)
        self.proba_scores = [n for m, n in self.model.predict_proba(self.x_test)]


    def scores(self):
        """
        Displays result of the model
        """
        print "Accuracy: %.2f%%" % (accuracy_score(self.y_test, self.y_pred) * 100)
        print "f1_score", f1_score(self.y_test, self.y_pred)
        print "precision_score", precision_score(self.y_test, self.y_pred)
        print "recall_score", recall_score(self.y_test, self.y_pred)
        print "cm matrix\n", confusion_matrix(self.y_test, self.y_pred)
        print "roc_auc_score", roc_auc_score(self.y_test, self.proba_scores)

    def plot(self):
        fpr, tpr, thresholds = roc_curve(self.y_test, self.proba_scores)
        plt.plot(fpr, tpr)
        plt.plot([(0, 0), (1, 1)])
        plt.show()