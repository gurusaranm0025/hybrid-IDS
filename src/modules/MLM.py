import joblib
import pandas as pd
import numpy as np
import warnings

from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ..config import ConfigClassFactory, Config
from ..ingest_data import DataIngestorFactory
from ..preprocessing import DataPreprocessor
from .DataSplitter import DataSplitter

warnings.filterwarnings("ignore")

SS_MLM_PARAMS = {
    'kernel': ['knn'],
    'gamma': [20, 1, 0.1, None],
    'n_neighbors': [3, 5]
}


class MLModel:
    def __init__(self, config: Config, param_grid: dict = SS_MLM_PARAMS, cv: int = 5, n_jobs: int = -1) -> None:
        self.config_: Config = config
        self._param_grid = param_grid
        self._cv = cv
        self._n_jobs = n_jobs
        
        self.model_: LabelPropagation = LabelPropagation()
        
        self._grid_srarch = GridSearchCV(estimator=self.model_, param_grid=self._param_grid, cv=self._cv, n_jobs=self._n_jobs)
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        print("\n --> TRAINING:")
        self._grid_srarch.fit(X.values, y.values) #.iloc[:, self.config_.SELECTED_FEATURES]
        
        self.model_ = self._grid_srarch.best_estimator_
        self.best_params_ = self._grid_srarch.best_params_
        
        print("\n The best score is ==> ", self._grid_srarch.best_score_)
        print("\n The best params is ==> \n", self.best_params_)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        print("\n --> PREDICTING:")
        return self.model_.predict(X) #.iloc[:, self.config_.SELECTED_FEATURES]
    
    def test(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        print("\n --> TESTING:")
        y_pred = self.predict(X)

        accuracy = accuracy_score(y_pred=y_pred, y_true=y)
        print(f"\n --> Accuracy: {(accuracy * 100.0):.4f}")
        print(f"\n --> Classification Report: \n{classification_report(y_true=y, y_pred=y_pred)}")
        print(f"\n --> Confusion Matrix: \n{confusion_matrix(y_pred=y_pred, y_true=y)}")
    
    def save_model(self):
        joblib.dump(self.model_, self.config_.MODEL_PATH_LP)
    
    def load_model(self, param_grid: dict = {}):
        self.model_ = joblib.load(self.config_.MODEL_PATH_LP)
        self._grid_srarch = GridSearchCV(estimator=self.model_, param_grid=param_grid, cv=self._cv, n_jobs=self._n_jobs)


if __name__ == "__main__":
    config = ConfigClassFactory.GetConfig('NBPO')
    
    ingestor = DataIngestorFactory.GetDataIngestor(config.DATASET_PATH)
    df = ingestor.ingest(config.DATASET_PATH, config.COLUMNS_TO_DROP)
    
    preprocess = DataPreprocessor(df, config.TARGET)
    preprocess.IPAddressEncoder(config.IP_COLUMNS)
    preprocess.MACAddressEncoder(config.MAC_COLUMNS)
    preprocess.target_col_val_modify(config.TARGET, config.TARGET_VAL_CHANGE)
    df = preprocess.getDataFrame()
    
    dataset = DataSplitter(df, config.TARGET)
    X_train, X_test, y_train, y_test = dataset.get_train_test()
    
    model = MLModel(config)
    
    print("\n ---> TRAINING MODE:")
    model.train(X_train, y_train)
    model.test(X_test, y_test)

    print("\n ---> LOADING MODE:")
    model.load_model()    
    model.test(X_test, y_test)
