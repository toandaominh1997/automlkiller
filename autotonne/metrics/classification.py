import numpy as np
from sklearn import metrics

class ClassificationMetricContainer():
    def __init__(self):
        super(ClassificationMetricContainer, self).__init__()
        self.report = {}

    def classification_report(self, y_true, y_pred):
        if 'accuracy_score' not in self.report.keys():
            self.report['accuracy_score'] = []
        self.report['accuracy_score'].append(metrics.accuracy_score(y_true, y_pred))

        if 'recall' not in self.report.keys():
            self.report['recall'] = []
        self.report['recall'].append(metrics.recall_score(y_true, y_pred))
        if 'precision' not in self.report.keys():
            self.report['precision'] = []
        self.report['precision'].append(metrics.precision_score(y_true, y_pred))
        if 'f1_score' not in self.report.keys():
            self.report['f1_score'] = []
        self.report['f1_score'].append(metrics.f1_score(y_true, y_pred))
        if 'cohen_kappa' not in self.report.keys():
            self.report['cohen_kappa'] = []
        self.report['cohen_kappa'].append(metrics.cohen_kappa_score(y_true, y_pred))

        if 'mcc' not in self.report.keys():
            self.report['mcc'] = []
        self.report['mcc'].append(metrics.matthews_corrcoef(y_true, y_pred))
    def classification_report_proba(self, y_true, y_pred_proba):
        if 'auc' not in self.report.keys():
            self.report['auc'] = []
        self.report['auc'].append(metrics.roc_auc_score(y_true, y_pred_proba))
    def score_mean(self):
        for key, value in self.report.items():
            self.report[key] = np.mean(value)
        return self.report
