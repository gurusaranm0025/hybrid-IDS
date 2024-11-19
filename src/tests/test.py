import pandas as pd
import sys

from ..config import config
from ..ingest_data import DataIngestorFactory

from .sampleData import GenSampleData
from .preprocessing import check_preprocessing


def check_column_drop(file_path: str, columns_to_drop: list) -> pd.DataFrame:
    """
        Checks DataIngestor.
        Checks whether unecessary columns are dropped or not.
        
        Parameters:
        filepath (string) - path to the dataset.
        columns_to_drop (list[string]) - columns to drop in the da
        
        Returns:
        (pandas.DataFrame) - the dataset.
    """    
    
    ingestor = DataIngestorFactory.GetDataIngestor(file_path)
    
    df = ingestor.ingest(file_path, columns_to_drop)
    
    print("DATA INGESTION WORKS PROPERLY")
    
    for col in config.COLUMNS_TO_DROP_ASNM_CDX:
        if col in df.columns:
            print("\n TEST FAILED.")
            print(f"{col} is present.")
            print(df.columns)
            raise ValueError(f"{col} is present.")
        
    print("ALL COLUMNS ARE DROPPED SUCCESSFULLY.")
    return df

if __name__ == "__main__":
    print("=>GENERATING SAMPLE DATA")
    GenSampleData()
    
    print("\n ===> FOR ASNM-CDX:")
    
    print("\n  > COLUMN DROP CHECKS:")
    df = check_column_drop(file_path=config.sample_ASNM_CDX, columns_to_drop=config.COLUMNS_TO_DROP_ASNM_CDX)
    
    print("\n  -Preprocessing check for ASNM-CDX")
    check_preprocessing(df, mac_columns=config.MAC_COLUMNS_CDX, ip_columns=config.IP_COLUMNS_CDX_TUN, target_col=config.target_ASNM_CDX)
    
    response = input("Are the data types okay [y/n] : ")
    if response == "y":
        print("CONTINUING THE TESTS")
    else:
        response = input("Do you wish to continue the test [y/n] : ")
        if not response == "y":
            sys.exit()
    
    print("\n ===> FOR ASNM-TUN:")
    
    print("\n  > COLUMN DROP CHECKS:")
    df = check_column_drop(file_path=config.sample_ASNM_TUN, columns_to_drop=config.COLUMNS_TO_DROP_ASNM_TUN)
    
    print("\n  -Preprocessing check for ASNM-TUN")
    check_preprocessing(df, ip_columns=config.IP_COLUMNS_CDX_TUN, target_col=config.target_ASNM_TUN, labels_to_change=config.target_val_TUN_NBPO)
    
    response = input("Are the data types okay [y/n] : ")
    if response == "y":
        print("CONTINUING THE TESTS")
    else:
        response = input("Do you wish to continue the test [y/n] : ")
        if not response == "y":
            sys.exit()
    
    print("\n ===> FOR ASNM-NBPO:")
    
    print("\n  > COLUMN DROP CHECKS:")
    df = check_column_drop(file_path=config.sample_ASNM_NBPO, columns_to_drop=config.COLUMNS_TO_DROP_ASNM_NBPO)
    
    print("\n  -Preprocessing check for ASNM-NBPO")
    check_preprocessing(df, ip_columns=config.IP_COLUMNS_NBPO, mac_columns=config.MAC_COLUMNS_NBPO, target_col=config.target_ASNM_NBPO, labels_to_change=config.target_val_TUN_NBPO)

    response = input("Are the data types okay [y/n] : ")
    if response == "y":
        print("CONTINUING THE TESTS")
    else:
        response = input("Do you wish to continue the test [y/n] : ")
        if not response == "y":
            sys.exit()