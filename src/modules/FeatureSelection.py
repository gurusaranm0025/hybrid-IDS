import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from .DataSplitter import DataSplitter

import pandas as pd

class FeatureSelectionBase(ABC):
            
    @abstractmethod
    def select(self, X: pd.DataFrame, y: pd.DataFrame) -> list:
        """
            Selects the best features from the given dataset.
            
            Parameters:
            X (pandas.DataFrame) : the dataframe which contains the input features.
            y (pandas.DataFrame) : the dataframe which contains the target feature.
            
            Returns:
            list: a list of the selected features position number.
        """
        pass
  
class ForwardFeatureSelection(FeatureSelectionBase):
    
    def select(X: pd.DataFrame, y: pd.DataFrame) -> list:
        model = LogisticRegression(max_iter=100)
        
        sfs = SequentialFeatureSelector(model, n_features_to_select=40, direction='forward', scoring='accuracy', cv=5)
        
        sfs.fit(X.values, y.values)
        
        features = sfs.get_support(indices=True)
        
        # print("Selected features: ", features)
        
        return features

class CorrelationAnalysis(FeatureSelectionBase):
    
    def select(X: pd.DataFrame, y: pd.DataFrame) -> list:

        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find columns with a correlation higher than a threshold
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]

        return to_drop
        # Drop these columns from the dataset
        # df_reduced = X.drop(columns=to_drop)

class StatisticalFeatureSelection(FeatureSelectionBase):
    def select(X: pd.DataFrame, y: pd.DataFrame) -> list:
        
        # Select the top k features based on ANOVA F-value or mutual information
        selector = SelectKBest(score_func=f_classif, k=80)  # Or mutual_info_classif
        
        X_selected = selector.fit_transform(X, y)
        return X_selected
        # selected_features = X.columns[selector.get_support()]

class PCAFeatureSelection(FeatureSelectionBase):
    def select(X: pd.DataFrame, y: pd.DataFrame) -> list:
        # Apply PCA to reduce dimensions while retaining 95% of the variance
        pca = PCA(n_components=0.9)
        X_pca = pca.fit_transform(X)
        return X_pca
        
class TreeBasedFeatureImportance(FeatureSelectionBase):
    
    def select( X: pd.DataFrame, y: pd.DataFrame) -> list:
        model = RandomForestClassifier()
        model.fit(X, y)

        # Get feature importances and select the top features
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
        top_features = feature_importances.nlargest(80).index
        return top_features
        # X_reduced = X[top_features]

class FeatureClassification:
    
    def __init__(self, df: pd.DataFrame, target_col: str, strategy: FeatureSelectionBase) -> None:
        self._df = df
        self._strategy = strategy
        self._target_col = target_col
        
    def select(self) -> list:
        """
            Perofrms the given strategy for feature selection.
            
            Parameters:
            None.
            
            Returns:
            (list) - a list of selected features.
        """
        
        X, y = DataSplitter(df=self._df, target_col=self._target_col).getXY()
        selected_features = self._strategy.select(X=X, y=y)
        return selected_features