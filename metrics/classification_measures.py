import numpy as np




def accuracy(y_true, y_pred):
    N = y_true.shape[0]
    correct_pred = np.sum(y_true == y_pred)
    return correct_pred / N


def recall(y_true, y_pred, class_id):
    correct_class_pred = np.sum(np.logical_and(y_true == y_pred, y_true == class_id))
    class_occurrences = np.sum(y_true == class_id)
    return correct_class_pred / class_occurrences

def precision(y_true, y_pred, class_id):
    correct_class_pred = np.sum(np.logical_and(y_true == y_pred, y_true == class_id))
    class_predictions = np.sum(y_pred == class_id)
    return correct_class_pred / class_predictions

def confusion_matrix(y_true, y_pred, normalize=None):
    number_of_labels = len(np.unique(y_true))
    cm = np.zeros((number_of_labels, number_of_labels), dtype=np.int32)
    for i in range(number_of_labels):
        for j in range(number_of_labels):
            cm[i,j] = np.sum(np.logical_and(y_true == i, y_pred == j), dtype=np.int32)
    return cm

def f_score(y_true, y_pred, class_id):
    _recall = recall(y_true, y_pred, class_id)
    _precision = precision(y_true, y_pred, class_id)
    return 2 * _precision*_recall / (_precision + _recall)
