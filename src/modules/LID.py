import numpy as np
import pandas as pd

from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from ..config import Config
# from ..ingest_data import DataIngestorFactory
# from .DataSplitter import DataSplitter

def compute_lid(distances):
    k = len(distances)
    r_max = np.max(distances)
    sum_log_ratios = np.sum(np.log(distances/r_max))
    
    lid_value = -(1/k)*sum_log_ratios
    return 1/lid_value

def compute_lid_df(X: np.ndarray, k: int) -> None:
    """
        Compute Local Intrinsic Dimensionality (LID) for each sample in X.
        
        Parameters:
        X (numpy.ndarray): np.ndarray of shape (n_samples, n_features)
        k (int): Number of nearest neighbors
        
        Returns: 
        np.ndarray of shape (n_samples,) containing LID for each sample.
    """
    
    dist_matrix = cdist(X, X, metric='euclidean')
    nearest_neighbors = np.sort(dist_matrix, axis=1)[:, 1:k+1]
    nn_df = pd.DataFrame(nearest_neighbors)
    lid_vals = nn_df.apply(compute_lid, axis=1)
    
    return pd.DataFrame(lid_vals, columns=['LID'])

def compute_lid_local_pca(X: np.ndarray, k: int):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    lid_vals = []
    for i in range(len(X)):
        pca = PCA()
        pca.fit(X[indices[i]])
        
        significant_components = np.sum(pca.explained_variance_ratio_ > 1e-5)
        lid_vals.append(significant_components)
    
    return np.array(lid_vals)

class LID:
    def __init__(self, config: Config) -> None:
        self.config_ = config
        self.dataset_: pd.DataFrame = pd.read_csv(self.config_.ACT_VALUES_DATASET_PATH)        
    
    def compute_lid_model(self, X: np.ndarray, k: int = None) -> pd.DataFrame:
        if k == None:
            k = self.config_.LID_K
        
        dist_matrix = cdist(X, X, metric='euclidean')
        nearest_neighbors = np.sort(dist_matrix, axis=1)[:, 1:k+1]
        nn_df = pd.DataFrame(nearest_neighbors)
        lid_vals = nn_df.apply(compute_lid, axis=1)
    
        return pd.DataFrame(lid_vals, columns=['LID'])
    
    def compute_lid_predict(self, X: pd.DataFrame) -> np.ndarray:
        X.columns = self.dataset_.columns
        X = pd.concat([self.dataset_, X])
        
        result = self.compute_lid_model(X)
        return result[self.dataset_.shape[0]:]
        