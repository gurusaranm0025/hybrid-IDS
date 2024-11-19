import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

from ..config import ConfigClassFactory
from ..ingest_data import DataIngestorFactory
from ..preprocessing import DataPreprocessor
from .DataSplitter import DataSplitter

class GaussianFieldsHarmonicFunction(BaseEstimator, ClassifierMixin):
    """
        Implements Semi-supervised learning with Gaussian Fields and harmonic function. From "Semi-Supervised Learning Using Gaussian Fields 
        and Harmonic Functions".  Xiaojin Zhu, Zoubin Ghahramani, John Lafferty.  
        The Twentieth International Conference on Machine Learning (ICML-2003).
    """
    
    def __init__(self, sigma: float = 1.0) -> None:
        self.sigma = sigma
            
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, labelled_data_percent: float = 0.1) -> None:
        """
            Fits the model with the training data.
            
            Parameters:
            ----------
            X (pandas.DataFrame) - Input features of training set.
            y (pandas.DataFrame) - Target feature of the training set.
            labelled_data_percent (float)[0.1 - 1.0] - float value determineds the amount of labelled data to be taken from the training set.
            
            Returns:
            -------
            None. 
        """
        self.X_train_ = X
        self.y_train_ = y
        
        self.labelled_indices_ = np.random.choice(X.shape[0], int(labelled_data_percent * X.shape[0]), replace=False)
        
        self.labels_train_ = self._prepare_data(y, self.labelled_indices_)
        
        self.L_ = self._compute_matrices(X)
        predicted_labels_train = self._gaussian_harmonic_functions(self.labels_train_)
        
        predicted_labels_train = np.argmax(predicted_labels_train, axis=1)
        accuracy = accuracy_score(y_pred=predicted_labels_train, y_true=self.y_train_)
        print(f"\n --> Accuracy: {(accuracy * 100.0):.4f}")
        print(f"\n --> Classification Report: \n{classification_report(y_true=y, y_pred=predicted_labels_train)}")
        print(f"\n --> Confusion Matrix: \n{confusion_matrix(y_pred=predicted_labels_train, y_true=y)}")

        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
            Predicts the target for the given input features.
            
            Parameters:
            ----------
            X (pandas.DataFrame) - Input features.
            
            Returns:
            -------
            (np.ndarray) - predictions for the input.
        """
        X_combined = np.vstack([self.X_train_, X])
    
        labels_combined = np.full((X_combined.shape[0], self.labels_train_.shape[1]), -1)
        labels_combined[:self.X_train_.shape[0]] = self.labels_train_
        
        L_combined = self._compute_matrices(X_combined)
        
        predicted_labels_combined = self._gaussian_harmonic_functions(labels_combined, L_combined)
    
        predictions_test = predicted_labels_combined[self.X_train_.shape[0]:]
        prediction_classes_test = np.argmax(predictions_test, axis=1)
    
        return prediction_classes_test

    def test(self, X: pd.DataFrame, y: pd.DataFrame, print_preds: bool = False) -> None:
        """
            Predicts the target feature for the given input features and performs basic benchmarking against the given actual target values.
            
            Parameters:
            ----------
            X (pandas.DataFrame) - Input features.
            y (pandas.DataFrame) - Actual target values.
            
            Returns:
            -------
            None. 
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y_true=y, y_pred=y_pred)
        
        print(f"\n --> Accuracy: {(accuracy * 100.0):.4f}")
        print(f"\n --> Classification Report: \n{classification_report(y_true=y, y_pred=y_pred)}")
        print(f"\n --> Confusion Matrix: \n{confusion_matrix(y_pred=y_pred, y_true=y)}")
        
        if print_preds:
            print(y_pred.tolist())
    
    def _prepare_data(self, y: pd.DataFrame | np.ndarray, labelled_indices: np.ndarray) -> np.ndarray:
        """
            Labelled and unlabelled data creation.
            
            Parameters:
            ----------
            y (pandas.DataFrame) - Target feature of the training set.
            labelled_indices (np.ndarray) - indices of the labelled observation in the dataset.
        """
        lb = LabelBinarizer()
        labels = lb.fit_transform(y)
        
        if labels.ndim == 1:
            labels.reshape(-1, 1)
        
        unlabelled_indices = np.setdiff1d(np.arange(len(y)), labelled_indices)
        labels[unlabelled_indices] = -1
        
        return labels
    
    def _compute_matrices(self, X: pd.DataFrame | np.ndarray) -> csr_matrix:
        """
            Computes the Laplacian Matrix for the given dataset's input features.
            
            Parameters:
            ----------
            X (pandas.DataFrame) - Input features.
            
            Returns:
            -------
            (scipy.sparse.csr_matrix) - for faster calculations, returns a csr_matrix
        """
        dists = squareform(pdist(X, 'euclidean'))
        
        W = np.exp(-dists**2 / (2 * self.sigma**2))
        np.fill_diagonal(W, 0)
        
        D = np.diag(W.sum(axis=1))
        
        L = csr_matrix(D - W)
        
        return L
    
    def _gaussian_harmonic_functions(self, labels: np.ndarray, L: csr_matrix = None) -> np.ndarray:
        """
            Gaussian Fields and Harmonic Functions.
            
            Parameters:
            ----------
            labels (np.ndarray) - it has the possible labels for each observations.
            L (csr_matrix) - the laplacian matrix.
            
            Returns:
            -------
            (np.ndarray) - the predictions made for the unlabelled data.
        """
        if L is None:
            L = self.L_
        
        labeled_indices = np.where(labels != -1)[0]
        unlabeled_indices = np.where(labels == -1)[0]
        
        L_uu = L[np.ix_(unlabeled_indices, unlabeled_indices)]
        L_ul = L[np.ix_(unlabeled_indices, labeled_indices)]
        
        Y_u: np.ndarray = -spsolve(L_uu, L_ul @ labels[labeled_indices])
        Y = labels.copy()
        
        Y_u = Y_u.reshape(-1, 1)
        
        Y[unlabeled_indices] = Y_u
        
        return Y

if __name__ == "__main__":
    config = ConfigClassFactory.GetConfig('NBPO')
    
    ingestor = DataIngestorFactory.GetDataIngestor(config.DATASET_PATH)
    df = ingestor.ingest(config.DATASET_PATH, config.COLUMNS_TO_DROP)
    
    preprocessor = DataPreprocessor(df, config.TARGET)
    preprocessor.IPAddressEncoder(config.IP_COLUMNS)
    preprocessor.MACAddressEncoder(config.MAC_COLUMNS)
    preprocessor.target_col_val_modify(config.TARGET, config.TARGET_VAL_CHANGE)
    df = preprocessor.getDataFrame()
    
    dataset = DataSplitter(df, config.TARGET)
    X_train, X_test, y_train, y_test = dataset.get_train_test()
    
    model = GaussianFieldsHarmonicFunction()
    model.fit(X_train, y_train, labelled_data_percent=0.3)
    
    model.test(X_test, y_test, True)