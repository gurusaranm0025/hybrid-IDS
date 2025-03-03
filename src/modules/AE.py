import joblib
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator
from typing import Literal

from ..config import Config, ConfigClassFactory
from ..ingest_data import DataIngestorFactory
from .DataSplitter import DataSplitter

warnings.filterwarnings("ignore")

class AEDetector:
    
    def __init__(self, config: Config, model_type: Literal['dense', 'conv1d', 'rnn'] ,estimator: BaseEstimator = None, param_grid: dict = {}, cv: int = 5, n_jobs: int = -1) -> None:
        self.config_ = config
        self._cv = cv
        self._n_jobs = n_jobs
        self._param_grid = param_grid
        self.model_type = model_type
        
        if estimator == None:
            self._model: SVC = SVC(kernel='rbf', C=0.1, random_state=config.RANDOM_STATE)
        else:
            self._model = estimator
        self.grid_search_ = GridSearchCV(estimator=self._model, param_grid=self._param_grid, cv=self._cv, n_jobs=self._n_jobs, verbose=2)
        self.best_params_: dict = None
        
        self.X_train_: pd.DataFrame = None
        self.y_train_: pd.DataFrame = None
        self.X_test_: pd.DataFrame = None
        self.y_test_: pd.DataFrame = None
        self._load_dataset()
    
    def set_model(self, estimator: BaseEstimator = None, param_grid: dict = {}, cv: int = None, n_jobs: int = None):
        if estimator != None:
            if cv != None:
                self._cv = cv
            if n_jobs != None:
                self._n_jobs = n_jobs
            self._param_grid = param_grid

            self._model = estimator
            self.grid_search_ = GridSearchCV(estimator=self._model, param_grid=self._param_grid, cv=self._cv, n_jobs=self._n_jobs, verbose=2)
        else:
            raise ValueError("No estimator value given.")
    
    def _load_dataset(self):
        ingestor = DataIngestorFactory.GetDataIngestor(self.config_.LID_DICT[self.model_type])
        df = ingestor.ingest(self.config_.LID_DICT[self.model_type], columns_to_drop=[], sep=',')
        dataset = DataSplitter(df, target_col=self.config_.LID_DATASET_TARGET)
        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = dataset.get_train_test()
                
    def train(self, X: pd.DataFrame = None, y: pd.DataFrame = None):
        print("\n --> TRAINING:")
        if X == None:
            print("\n ->AE Detector:>Training Using default value for training set")
            X = self.X_train_
        if y == None:
            print("\n ->AE Detector:>Training Using default value for testing set")
            y = self.y_train_
            
        self.grid_search_.fit(X.values, y.values)
        self._model = self.grid_search_.best_estimator_
        self.best_params_ = self.grid_search_.best_params_
        
        print("\n The best score is ==> ", self.grid_search_.best_score_)
        print("\n The best params is ==> \n", self.best_params_)
    
    def predict(self, X: pd.Series | np.ndarray) -> np.ndarray:
        print("\n --> PREDICTING:")
        return self._model.predict(X)
        
    def test(self, X: pd.DataFrame = None, y: pd.DataFrame = None) -> None:
        print("\n --> TESTING:")
        """
            Test the trained model with the given test dataset, and prints the outcome.
            
            Parameters:
            X_test (pandas.DataFrame) - input features for the test.
            y_test (pandas.DataFrame) - expected output or target values of the input features.
            
            Returns:
            None. It prints the output.
        """
        if X == None:
            print("\n ->AE Detector: Using default value for training set")
            X = self.X_test_
        if y == None:
            print("\n ->AE Detector: Using default value for testing set")
            y = self.y_test_
        
        y_pred = self._model.predict(X.values)
        
        accuracy = accuracy_score(y_true=y.values, y_pred=y_pred)
        print(f"\n ACCURACY SCORE : {(accuracy * 100.0):.4f}")
        print("\n CLASSIFICATION REPORT : \n", classification_report(y, y_pred))
        print("\n CONFUSION MATRIX : \n", confusion_matrix(y, y_pred))
    
    def save_model(self):
        joblib.dump(self._model, self.config_.AE_DICT[self.model_type])
    
    def load_model(self, param_grid: dict = None, cv: int = None):
        if param_grid != None:
            self._param_grid = param_grid
        if cv != None:
            self._cv = cv
                        
        self._model = joblib.load(self.config_.AE_DICT[self.model_type])
        self.grid_search_ = GridSearchCV(estimator=self._model, param_grid=self._param_grid, cv=self._cv, n_jobs=self._n_jobs, verbose=2)

AE_MODELS = {
    # 'RFC': [RandomForestClassifier(random_state=Config.RANDOM_STATE), {
    #     'n_estimators': [100, 200, None], 
    #     'max_depth': [10, 20, None], 
    #     'min_samples_split': [2, 5, None], 
    #     'min_samples_leaf': [1, 2, None], 
    #     'max_features': ['sqrt', 'log2', None]
    #     }],
    # 'GBC': [GradientBoostingClassifier(random_state=Config.RANDOM_STATE), {
    #     'n_estimators': [100, 200, None],
    #     'learning_rate': [0.01, 0.1, 0.2, None],
    #     'max_depth': [3, 5, None],
    #     'min_samples_split': [2, 5, None],
    #     'min_samples_leaf': [1, 2, 4, None],
    #     'subsample': [0.8, 1.0, None]
    #     }],
    # 'LR': [LogisticRegression(random_state=Config.RANDOM_STATE), {
    #     'C': [0.01, 0.1, 1, 10, 100],
    #     'penalty': ['l1', 'l2', None],
    #     'solver': ['liblinear', 'saga', None]
    #     }],
    # 'DTC': [DecisionTreeClassifier(random_state=Config.RANDOM_STATE), {
    #     'max_depth': [None, 10, 20, 30, 40],
    #     'min_samples_split': [2, 5, 10, 20, None],
    #     'min_samples_leaf': [1, 2, 4, 10, None],
    #     'criterion': ['gini', 'entropy']
        # }],
    'SVC': [SVC(random_state=Config.RANDOM_STATE), {
        'C': [0.1],
        'kernel': ['rbf'], # linear
        'gamma': ['scale', 'auto'],  # scale
        # 'degree': [2, 3, None]  # Only relevant for 'poly' kernel
        }]
}

if __name__ == "__main__":
    ae_detector = AEDetector(config=ConfigClassFactory.GetConfig("NBPO"))
    print("\n ---> TRAINING MODE:")
    ae_detector.train()
    ae_detector.test()
    print("\n ---> LOADING MODE:")
    ae_detector.load_model()
    ae_detector.test()
    
