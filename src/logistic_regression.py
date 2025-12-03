import numpy as np
import numpy.typing as npt
import pickle
from typing import Literal, Optional

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]

class LogisticRegressionModel:
    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000, multi_class: Literal['ovr', 'multinomial'] = 'ovr'):
        self.lr: float = learning_rate
        self.epochs: int = n_iters
        self.multi_class: Literal['ovr', 'multinomial'] = multi_class
        self.weights: Optional[FloatArray] = None
        self.bias = None
        self.classes = None
        self.losses = []
        self.history_params = []

    def _sigmoid(self, x: FloatArray) -> FloatArray:
        return 1 / (1 + np.exp(-x))
    
    def _softmax(self, x: FloatArray) -> FloatArray:
        # TODO: Implementasi fungsi softmax
        pass

    def fit(self, X: FloatArray, y: FloatArray):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        if self.multi_class == 'ovr':
            self._fit_ovr(X, y, n_samples, n_features, n_classes)
        else:
            self._fit_softmax(X, y, n_samples, n_features, n_classes)
    
    def _fit_ovr(self, X: FloatArray, y: FloatArray, n_samples: int, n_features: int, n_classes: int) -> None:
        # TODO: Implementasi One-vs-Rest LogisticRegression Model
        pass

    def _fit_softmax(self, X: FloatArray, y: FloatArray, n_samples: int, n_features: int, n_classes: int) -> None:
        # TODO: Implementasi Softmax Regression
        pass

    def predict(self, X: FloatArray):
        # TODO: Implementasi prediksi multikelas
        pass

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)