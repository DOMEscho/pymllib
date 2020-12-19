import unittest
from sklearn import metrics as sk_metrics
import numpy as np
from metrics import confusion_matrix, accuracy, recall, precision, f_score

class TestClassificationMeasures(unittest.TestCase):
     
    y_true = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0])
    y_pred = np.array([0,0,0,1,1,1,1,1,0,0,0,1,1])


    def test_confusion_matrix(self):
        expected_cm = sk_metrics.confusion_matrix(self.y_true, self.y_pred) 
        actual_cm = confusion_matrix(self.y_true, self.y_pred)
        np.testing.assert_allclose(actual_cm, expected_cm, atol=1e-8)
        

    def test_accuracy(self):
        expected_accuracy = sk_metrics.accuracy_score(self.y_true, self.y_pred)
        actual_accuracy = accuracy(self.y_true, self.y_pred)

        self.assertEqual(actual_accuracy, expected_accuracy)


    def test_recall(self):
        expected_recall = sk_metrics.recall_score(self.y_true, self.y_pred)
        actual_recall = recall(self.y_true, self.y_pred, 1)

        self.assertEqual(actual_recall, expected_recall)
        

    def test_precision(self):
        expected_precision = sk_metrics.precision_score(self.y_true, self.y_pred)
        actual_precision = precision(self.y_true, self.y_pred, 1)
        self.assertEqual(actual_precision, expected_precision)
   
    def test_f_score(self):
        precision = sk_metrics.precision_score(self.y_true, self.y_pred)
        recall = sk_metrics.recall_score(self.y_true, self.y_pred)
        expected_fscore = 2 * precision * recall / (precision + recall)
        actual_fscore = f_score(self.y_true, self.y_pred, 1)
        self.assertEqual(actual_fscore, expected_fscore)



if __name__ == '__main__':
    unittest.main()
