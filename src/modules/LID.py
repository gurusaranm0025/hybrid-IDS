import numpy as np
import pandas as pd

from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from typing import Literal

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

# class LID:
#     def __init__(self, config: Config, model_type: Literal['dense', 'conv1d', 'rnn'] = 'dense') -> None:
#         self.config_ = config
#         self.model_type_ = model_type
#         self.dataset_: pd.DataFrame = pd.read_csv(self.config_.ACT_VALS_DICT[model_type])        
    
#     def compute_lid_model(self, X: np.ndarray, k: int = None) -> pd.DataFrame:
#         if k == None:
#             k = self.config_.LID_K
        
#         dist_matrix = cdist(X, X, metric='euclidean')
#         nearest_neighbors = np.sort(dist_matrix, axis=1)[:, 1:k+1]
#         nn_df = pd.DataFrame(nearest_neighbors)
#         lid_vals = nn_df.apply(compute_lid, axis=1)
    
#         return pd.DataFrame(lid_vals, columns=['LID'])
    
#     def compute_lid_predict(self, X: pd.DataFrame) -> np.ndarray:
#         X.columns = self.dataset_.columns
#         X = pd.concat([self.dataset_, X])
        
#         result = self.compute_lid_model(X)
#         return result[self.dataset_.shape[0]:]

# from sklearn.neighbors import NearestNeighbors
# import numpy as np
# import pandas as pd

class LID:
    def __init__(self, config: Config, model_type: Literal['dense', 'conv1d', 'rnn'] = 'dense') -> None:
        self.config_ = config
        self.model_type_ = model_type
        self.dataset_: pd.DataFrame = pd.read_csv(self.config_.ACT_VALS_DICT[model_type])
        self.k_ = self.config_.LID_K
        
        # Precompute nearest neighbors for the dataset
        self.nbrs_ = NearestNeighbors(n_neighbors=self.k_, metric='euclidean').fit(self.dataset_.values)
    
    def compute_lid_single(self, distances: np.ndarray) -> float:
        """
        Compute LID for a single point using distances to its neighbors.
        """
        if len(distances) < 2:
            return 0.0  # Avoid log(0) error for isolated points
        distances = np.sort(distances)
        return -1 / np.mean(np.log(distances[1:] / distances[0]))
    
    def compute_lid_model(self, X: np.ndarray) -> pd.DataFrame:
        """
        Compute LID for all points in X relative to the dataset.
        """
        print("\n -->calcing lid vals for models")
        # Find nearest neighbors in the dataset for X
        distances, _ = self.nbrs_.kneighbors(X, n_neighbors=self.k_)
        lid_vals = [self.compute_lid_single(dist) for dist in distances]
        
        return pd.DataFrame(lid_vals, columns=['LID'])
    
    def compute_lid_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute LID for new data points in X.
        """
        print("\n -->calcing lid vals")
        # Align columns
        X.columns = self.dataset_.columns
        return self.compute_lid_model(X.values).fillna(0).to_numpy()
