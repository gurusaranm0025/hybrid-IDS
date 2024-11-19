import pandas as pd

from ..ingest_data import DataIngestorFactory
from ..preprocessing import DataPreprocessor

from ..modules.FeatureSelection import FeatureClassification, CorrelationAnalysis, StatisticalFeatureSelection, TreeBasedFeatureImportance, PCAFeatureSelection, ForwardFeatureSelection
from ..config import config

def feature_selection_check(df: pd.DataFrame, target_col: str) -> None:
    """
        Runs tests for checking feature selection.
        
        Parameters:
        df (pandas.DataFrame) - the dataset for running checks.
        target_col (string) - name of the target column
        
        Returns:
        None. Prints the output.
    """
    fs = FeatureClassification(df=df, strategy=ForwardFeatureSelection, target_col=target_col)
    selected_features = fs.select()
    
    print("     --Selcted features ==> ", selected_features)

if __name__ == "__main__":
    print("\n -Running tests for Feature Selection:")
    
    ingestor = DataIngestorFactory.GetDataIngestor(config.ASNM_CDX)

    print("\n -For ASNM-CDX:")
    
    df = ingestor.ingest(file_path=config.ASNM_CDX, columns_to_drop=config.COLUMNS_TO_DROP_ASNM_CDX)
    preprocessor = DataPreprocessor(df=df, target_col=config.target_ASNM_CDX)
    preprocessor.IPAddressEncoder(cols=config.IP_COLUMNS_CDX_TUN)
    preprocessor.MACAddressEncoder(cols=config.MAC_COLUMNS_CDX)
    df = preprocessor.getDataFrame()
    
    feature_selection_check(df=df, target_col=config.target_ASNM_CDX)

    print("\n -For ASNM-TUN:")
    
    df = ingestor.ingest(file_path=config.ASNM_TUN, columns_to_drop=config.COLUMNS_TO_DROP_ASNM_TUN)
    preprocessor = DataPreprocessor(df=df, target_col=config.target_ASNM_TUN)
    preprocessor.IPAddressEncoder(cols=config.IP_COLUMNS_CDX_TUN)
    df = preprocessor.getDataFrame()
    
    feature_selection_check(df=df, target_col=config.target_ASNM_TUN)

    print("\n -For ASNM-NBPO:")
    
    df = ingestor.ingest(file_path=config.ASNM_NBPO, columns_to_drop=config.COLUMNS_TO_DROP_ASNM_NBPO)
    preprocessor = DataPreprocessor(df=df, target_col=config.target_ASNM_NBPO)
    preprocessor.IPAddressEncoder(cols=config.IP_COLUMNS_NBPO)
    preprocessor.MACAddressEncoder(cols=config.MAC_COLUMNS_NBPO)
    df = preprocessor.getDataFrame()
    
    feature_selection_check(df=df, target_col=config.target_ASNM_NBPO)
