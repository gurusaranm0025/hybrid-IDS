from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import ipaddress

# from training.DataSplitter import DataSplitter

# a class to preprocess the dataset
class DataPreprocessor:
    
    def __init__(self, df: pd.DataFrame, target_col: str) -> None:
        """
            Initialises the Data Preprocessor for preprocessing the datasets.
            
            Parameters:
            df (pandas.DataFrame) - dataset for preprocessing.
            
            Returns:
            The DataPreprocessor class.
        """
        self._df: pd.DataFrame = df
        self._target_col = target_col
        self._scale_int_float_cols()
        self._bool_col_change()
        
    def _scale_int_float_cols(self) -> None:
        """
            It scales the ineger and float columns.
            
            Parameters:
            None.
            
            Returns:
            None.
        """
        self._df.dropna(inplace=True)
        num_cols = self._df.select_dtypes(include=['int', 'float64']).columns
        
        if self._target_col in num_cols.to_list():
            num_cols = num_cols.drop(self._target_col)
        
        scaler = StandardScaler()
        self._df[num_cols] = scaler.fit_transform(self._df[num_cols])
        
    def _bool_col_change(self) -> None:
        """
            Identifies boolean columns and changes it into integer columns
            
            Parameters:
            None.
            
            Returns:
            None.
        """
        bool_cols = self._df.select_dtypes(include='bool').columns
        self._df[bool_cols] = self._df[bool_cols].astype(int)
            
    
    def IPAddressEncoder(self, cols: list[str]) -> None:
        """
            Encodes the IP addresses in the dataset into numerical and scales them.
            
            Parameters:
            ncols (list) - the name of the columns to preprocess.
            
            Returns:
            None.
        """
        
        for col in cols:
            ip_mappings = {ip: int(ipaddress.IPv4Address(ip)) for ip in self._df[col].unique()}
            
            self._df[col] = self._df[col].map(ip_mappings)
        
        scaler = MinMaxScaler()
        self._df[cols] = scaler.fit_transform(self._df[cols])
        
        return self._df
    
    def MACAddressEncoder(self, cols: list[str] | None) -> None:
        """
            Encodes the MAC addresses in the the dataframe into numerical data and scales them.
            
            Parameters:
            cols (list[str]) - the column names to preprocess.
            
            Returns:
            None.
        """
        if cols == None or cols == []:
            return
        
        def mac_to_int(mac: str):
            return int(mac.replace(":", ""), 16)
        
        for col in cols:
            self._df[col] = self._df[col].apply(mac_to_int)
        
        scaler = MinMaxScaler()
        self._df[cols] = scaler.fit_transform(self._df[cols])
        
        return self._df
    
    def target_col_val_modify(self, target_col: str, label_intrsuct: list) -> None:
        """
            Replaces the label present in the column with the given new label.
            
            Parametres:
            target_col (string) - name of the target column
            label_instruct (list[list[any, any]]) - contains list of current label with the new label to replace
            
            Returns:
            None.
        """
        for old_label, new_label in label_intrsuct:
            self._df[target_col] = np.where(self._df[target_col] == old_label, new_label, self._df[target_col])
    
    def col_val_mod(self, target_col: str, old_label, new_label) -> None:
        """
            Combines the harmful traffic labels into one label, for ASNM-TUN, ASNM-NBPO.
            
            Parametres:
            target_col (string) - name of the target column
            old_label (any) -  current label to change
            new_label (any) - new label to replace the old label
            
            Returns:
            None.
        """
        self._df[target_col] = np.where(self._df[target_col] == old_label, new_label, self._df[target_col])
        
        return self._df
    
    def getDataFrame(self) -> pd.DataFrame:
        """
            Returns the preprocessed Data Frame.
            
            Parmaeters:
            None.
            
            Returns:
            The preprocessed Data Frame. 
        """
        return self._df