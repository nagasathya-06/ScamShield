"""
From-Scratch Implementation of Logistic Regression with L2 Regularization using NumPy.
Includes Binary Cross-Entropy loss computation, gradient descent optimization,
loss tracking, and probability calibration.
"""

import numpy as np
from scipy.sparse import issparse
from typing import List, Optional


class ScratchLogisticRegression:
    """
    Binary Logistic Regression Classifier implemented from scratch using NumPy.
    
    Parameters:
    -----------
    learning_rate : float, default=0.1
        Step size for gradient descent optimization.
    n_iters : int, default=300
        Maximum number of training iterations.
    l2_reg : float, default=0.1
        L2 regularization strength (lambda / C penalty).
    optimizer : str, default='adam'
        Optimization method ('adam' or 'gd'). Adam converges rapidly on high-dimensional text data.
    tolerance : float, default=1e-5
        Early stopping tolerance based on change in loss.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        n_iters: int = 400,
        l2_reg: float = 0.01,
        optimizer: str = 'adam',
        tolerance: float = 1e-6
    ):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.l2_reg = l2_reg
        self.optimizer = optimizer
        self.tolerance = tolerance
        
        self.weights_ = None
        self.bias_ = None
        self.loss_history_: List[float] = []
        self.classes_ = np.array([0, 1])

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid function:
        Avoids overflow for large negative or positive values.
        """
        z_clipped = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _compute_loss(self, y: np.ndarray, y_pred_prob: np.ndarray) -> float:
        """
        Binary Cross-Entropy Loss with L2 regularization penalty:
        J = - (1/m) * sum(y*log(p) + (1-y)*log(1-p)) + (lambda / (2*m)) * ||w||^2
        """
        m = len(y)
        eps = 1e-15
        p = np.clip(y_pred_prob, eps, 1.0 - eps)
        bce = - (1.0 / m) * np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        l2_penalty = (self.l2_reg / (2.0 * m)) * np.sum(self.weights_ ** 2)
        return float(bce + l2_penalty)

    def fit(self, X, y):
        """
        Fits logistic regression model using gradient descent or Adam.
        
        Parameters:
        -----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) binary labels 0 or 1
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape

        # Initialize weights with Xavier/Glorot uniform initialization
        limit = np.sqrt(6.0 / (n_samples + n_features))
        self.weights_ = np.random.uniform(-limit, limit, size=(n_features, 1))
        self.bias_ = 0.0
        self.loss_history_ = []

        # Adam optimizer state variables
        m_w = np.zeros_like(self.weights_)
        v_w = np.zeros_like(self.weights_)
        m_b = 0.0
        v_b = 0.0
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        prev_loss = float('inf')

        for i in range(1, self.n_iters + 1):
            # 1. Forward Pass: z = Xw + b
            if issparse(X):
                z = X.dot(self.weights_).ravel() + self.bias_
            else:
                z = np.dot(X, self.weights_).ravel() + self.bias_

            y_pred_prob = self._sigmoid(z)

            # 2. Compute Loss
            current_loss = self._compute_loss(y, y_pred_prob)
            self.loss_history_.append(current_loss)

            # Check early stopping tolerance
            if abs(prev_loss - current_loss) < self.tolerance:
                break
            prev_loss = current_loss

            # 3. Compute Gradients
            error = (y_pred_prob - y).reshape(-1, 1)  # shape (m, 1)
            
            if issparse(X):
                # Gradient of L2-regularized BCE w.r.t weights
                dw = (X.T.dot(error) / n_samples) + (self.l2_reg / n_samples) * self.weights_
            else:
                dw = (np.dot(X.T, error) / n_samples) + (self.l2_reg / n_samples) * self.weights_
                
            db = float(np.mean(error))

            # 4. Weight Updates
            if self.optimizer == 'adam':
                m_w = beta1 * m_w + (1 - beta1) * dw
                v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
                m_w_hat = m_w / (1.0 - beta1 ** i)
                v_w_hat = v_w / (1.0 - beta2 ** i)
                self.weights_ -= (self.learning_rate / (np.sqrt(v_w_hat) + eps)) * m_w_hat

                m_b = beta1 * m_b + (1 - beta1) * db
                v_b = beta2 * v_b + (1 - beta2) * (db ** 2)
                m_b_hat = m_b / (1.0 - beta1 ** i)
                v_b_hat = v_b / (1.0 - beta2 ** i)
                self.bias_ -= (self.learning_rate / (np.sqrt(v_b_hat) + eps)) * m_b_hat
            else:
                # Standard Gradient Descent
                self.weights_ -= self.learning_rate * dw
                self.bias_ -= self.learning_rate * db

        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Probability estimation for binary classes: [P(Safe=0), P(Scam=1)]
        """
        if issparse(X):
            z = X.dot(self.weights_).ravel() + self.bias_
        else:
            z = np.dot(X, self.weights_).ravel() + self.bias_

        prob_scam = self._sigmoid(z)
        prob_safe = 1.0 - prob_scam
        return np.column_stack([prob_safe, prob_scam])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """
        Binary prediction based on probability threshold.
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def get_coefficients(self) -> np.ndarray:
        """Returns the weight vector."""
        return self.weights_.ravel()
