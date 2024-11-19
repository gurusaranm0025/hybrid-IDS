import pandas as pd

from ..preprocessing import DataPreprocessor

def check_preprocessing(df: pd.DataFrame, target_col: str = "", labels_to_change: list = [], ip_columns: list[str] = [], mac_columns: list[str] = []) -> None:
    """
        Checks data preprocessing.
        
        Parametrers:
        df (pandas.DataFrame) - dataset to preprocess.
        ip_columns (list[string]) - IP Adress columns to preprocess.
        mac_columns (list[string]) - MAC Adress columns to preprocess.
                
        Returns:
        None. It prints the output.
    """
    
    preprocessor = DataPreprocessor(df, target_col)
    
    print("\n -Encoding IP Addresses:")
    
    if len(ip_columns) > 0:
        print("\n Data type before preprocessing ==> \n", df[ip_columns].dtypes)

        df = preprocessor.IPAddressEncoder(ip_columns)

        print("\n Data type after preprocessing ==> \n", df[ip_columns].dtypes)
    else:
        print("     No IP Address columns given.")    
    
    
    print("\n -Encoding MAC Addresses:")
    
    if len(mac_columns) > 0:
        print("\n Data type before preprocessing ==> \n", df[mac_columns].dtypes)

        df = preprocessor.MACAddressEncoder(mac_columns)

        print("\n Data type after preprocessing ==> \n", df[mac_columns].dtypes)
    else:
        print("     No MAC Address columns given.")
    
    print("\n -Checking for Boolean Columns:")
    bool_cols = df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        print("     Boolean columns are present in the dataset.")
        print(bool_cols)
    else:
        print("     No boolean columns are present.")
    
    print("\n -Combining the labels")
    if len(labels_to_change)>0:
        
        for labels in labels_to_change:
            print(f"\n --Changing the {labels[0]} to {labels[1]} in column ==> {target_col}")
            df = preprocessor.col_val_mod(target_col, labels[0], labels[1])
        
        print(f"    Unique values of column {target_col} ==> {df[target_col].unique()}, and its data type ==> {df[target_col].dtype}")
    else:
        print("     skipping label combining.")
        
    print("\n -Value counts in the target column:")
    if len(target_col) > 0:
        print(df[target_col].value_counts().to_dict())
        print(f"\n    Unique values of column {target_col} ==> {df[target_col].unique()}, and its data type ==> {df[target_col].dtype}")
    
    print("\n -Other type of columns in the dataset:")
    print(df.select_dtypes(include=['bool', 'object']).columns.to_list())
        