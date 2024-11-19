import pandas as pd

from ..ingest_data import DataIngestorFactory
from ..config import config

def unique_cols() -> list:
    """
        Identifies the unique columns of the three datasets.
        
        Paramteters:
        None.
        
        Returns:
        (list) - list of three items containing the unique columns in the three datasets in the order (CDX, TUN, NBPO). 
    """
    
    ingestor = DataIngestorFactory.GetDataIngestor(config.sample_ASNM_CDX)
    df1 = ingestor.ingest(file_path=config.sample_ASNM_CDX, columns_to_drop=config.COLUMNS_TO_DROP_ASNM_CDX)
    df2 = ingestor.ingest(file_path=config.sample_ASNM_TUN, columns_to_drop=config.COLUMNS_TO_DROP_ASNM_TUN)
    df3 = ingestor.ingest(file_path=config.sample_ASNM_NBPO, columns_to_drop=config.COLUMNS_TO_DROP_ASNM_NBPO)
    
    df1_cols = set(df1.columns)
    df2_cols = set(df2.columns)
    df3_cols = set(df3.columns)
    
    uniq_col_df1 = (df1_cols - df2_cols) - df3_cols
    uniq_col_df2 = (df2_cols - df1_cols) - df3_cols
    uniq_col_df3 = (df3_cols - df1_cols) - df2_cols
    
    return [uniq_col_df1, uniq_col_df2, uniq_col_df3]

if __name__ == "__main__":
    results = unique_cols()
    for idx, result in enumerate(results):
        print(f"\n\n\n The unique columns in dataset {idx+1} ==> {result}")