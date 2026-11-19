"""
Automated Unit Tests for From-Scratch Algorithms and Sklearn Equivalence.
Asserts mathematical consistency, shape integrity, and numerical stability.
"""

import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from src.models.scratch_naive_bayes import ScratchNaiveBayes
from src.models.scratch_logistic_regression import ScratchLogisticRegression


class TestScratchModels(unittest.TestCase):

    def test_scratch_naive_bayes_parity(self):
        """
        Verifies that ScratchNaiveBayes produces identical class probabilities
        and predictions compared to sklearn.naive_bayes.MultinomialNB.
        """
        np.random.seed(42)
        X = np.random.randint(0, 5, size=(100, 20)).astype(float)
        y = np.random.choice([0, 1], size=100)

        # Scratch model
        scratch_nb = ScratchNaiveBayes(alpha=1.0)
        scratch_nb.fit(X, y)
        scratch_preds = scratch_nb.predict(X)
        scratch_probs = scratch_nb.predict_proba(X)

        # Sklearn model
        sklearn_nb = MultinomialNB(alpha=1.0)
        sklearn_nb.fit(X, y)
        sklearn_preds = sklearn_nb.predict(X)
        sklearn_probs = sklearn_nb.predict_proba(X)

        # Assert 100% prediction match
        np.testing.assert_array_equal(scratch_preds, sklearn_preds)
        # Assert probability estimates match closely
        np.testing.assert_allclose(scratch_probs, sklearn_probs, atol=1e-5)

    def test_scratch_logistic_regression_convergence(self):
        """
        Verifies that ScratchLogisticRegression minimizes cross-entropy loss
        and achieves high accuracy.
        """
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = (X[:, 0] + X[:, 1] * 2.0 > 0).astype(int)

        scratch_lr = ScratchLogisticRegression(learning_rate=0.1, n_iters=250, l2_reg=0.01)
        scratch_lr.fit(X, y)

        self.assertLess(scratch_lr.loss_history_[-1], scratch_lr.loss_history_[0])
        self.assertLess(scratch_lr.loss_history_[-1], 0.35)

        preds = scratch_lr.predict(X)
        acc = np.mean(preds == y)
        self.assertGreater(acc, 0.90)

    def test_numerical_stability(self):
        """
        Verifies that sigmoid does not overflow or return NaN on extreme values.
        """
        lr = ScratchLogisticRegression()
        large_z = np.array([-1000.0, 0.0, 1000.0])
        sig = lr._sigmoid(large_z)
        self.assertFalse(np.isnan(sig).any())
        self.assertTrue(np.all(sig >= 0.0) and np.all(sig <= 1.0))


if __name__ == '__main__':
    unittest.main()
