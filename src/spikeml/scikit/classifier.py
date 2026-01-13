import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from spikeml.datasets.dataset import SimpleDataset, DataLoader
from spikeml.core.runner import DataRunner, InferenceRunner
from spikeml.datasets.encoder import one_hot


class ScikitClassifierAdapter(BaseEstimator, ClassifierMixin):
    def __init__(self, ref, epochs=1, labels=None, batch_size=-1):
        self.ref = ref
        self.epochs = epochs
        self.labels = labels
        self.batch_size = batch_size
        
    def fit(self, X, y):
        # Validate inputs
        X, y = check_X_y(X, y)

        # Store classes seen during fit
        self.classes_ = unique_labels(y)
        
        n = len(self.labels if self.labels is not None else self.classes_)
        y_ = one_hot(y, n)
        dataset = SimpleDataset(X, y_)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        runner = DataRunner(self.ref, loader, epochs=self.epochs, plot=False, log_step=-1, log_epoch=-1, callback=None)
        results_ = runner.run(options={})

        return self  # important!

    def _predict(self, X):
        #check_is_fitted(self, ["classes_"])
        X = check_array(X)
        dataset = SimpleDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        runner = InferenceRunner(self.ref, loader, log_step=-1)
        yy_prob = runner.run(options={})
        yy_pred = np.argmax(yy_prob, -1)
        return yy_prob, yy_pred
        
    def predict(self, X):
        _, yy_pred = self._predict(X)
        return yy_pred
    
    def predict_proba(self, X):
        yy_prob, _ = self._predict(X)
        return yy_prob


def test_classifier():
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(ScikitClassifierAdapter())
