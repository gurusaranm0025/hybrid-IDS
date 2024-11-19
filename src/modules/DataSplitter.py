from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTE

def Split(df: pd.DataFrame, target_feature: str) -> list[pd.DataFrame]:
    """
        Splits the dataset into X (input features), y (target feature)
            
        Parameters:
        df (pandas.DataFrame) - the dataset to split.
        target_feature (str) - name of the target feature.
            
        Returns:
        X (pandas.DataFraeme) - input features data frame
        y (pandas.DataFraeme) - target feature data frame
    """
    return [df.drop(columns=[target_feature], axis=1), df[target_feature]]
        
    
def Train_Test_Split(df:pd.DataFrame, target_feature: str, test_size: float = 0.2, random_state: int = 0) -> list[pd.DataFrame]:
    """
        Splits the given dataset into training and testing set, while removing preconfigured columns.
            
        Parameters:
        df (pandas.DataFrame) - the dataset to split.
        target_feature (str) - name of the target feature.
        test_size (float) - value indicating the split raio for training and testing set.
        random_state (int) - Random state.
            
        Returns:
        X_train (pandas.DataFrame) - input features for training the model.
        y_train (pandas.DataFrame) - target feature for training the model.
        X_test (pandas.DataFrame)- input featuers for testing the model.
        y_test (pandas.DataFrame)- target for testing the model.
    """
        
    X = df.drop(columns=[target_feature], axis=1)
    y = df[target_feature]
        
    # FINISH THE REST OF THE TODOS THEN CONTINUE HERE.
    return train_test_split(X, y, test_size=test_size, stratify=y,random_state=random_state)

class DataSplitter():
    
    def __init__(self, df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 0) -> None:
        self.DF = df
        self.smote_ = SMOTE(k_neighbors=10, random_state=42)
        
        self.X, self.y = Split(self.DF, target_col)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, stratify=self.y ,test_size=test_size, random_state=random_state)

        self.X_train_resampled, self.y_train_resampled = self.smote_.fit_resample(self.X_train, self.y_train)
                
        self.X.reset_index(drop=True, inplace=True)
        self.y.reset_index(drop=True, inplace=True)
        self.X_train.reset_index(drop=True, inplace=True)
        self.X_test.reset_index(drop=True, inplace=True)
        self.y_train.reset_index(drop=True, inplace=True)
        self.y_test.reset_index(drop=True, inplace=True)
        self.X_train_resampled.reset_index(drop=True, inplace=True)
        self.y_train_resampled.reset_index(drop=True, inplace=True)
    
    def get_resampled_dataset(self) -> list[pd.DataFrame]:
        X, y = self.smote_.fit_resample(self.X, self.y)
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        return [X, y]
    
    def getXY(self) -> list[pd.DataFrame]:
        """            
            Returns a list contianing the X and y partitions of the dataset.
        """
        return [self.X, self.y]
    
    def get_test(self):
        """
            Returns the testing sets.
        """
        return [self.X_test, self.y_test]
    
    def get_train(self):
        """
            Returns the training set.
        """
        return [self.X_train, self.y_train]
    
    def get_train_resampled(self):
        """
            Returns the SMOTE applied training sets
        """
        return [self.X_train_resampled, self.y_train_resampled]
    
    def get_train_test(self) -> list[pd.DataFrame]:
        """
            Returns a list contianing the X_train, X_test, y_trian, y_test partitions of the dataset.
        """
        return [self.X_train, self.X_test, self.y_train, self.y_test]