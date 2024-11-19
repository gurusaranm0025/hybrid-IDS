import pandas as pd

from ..config import config

def GenSampleData() -> pd.DataFrame:
    """
        Gives a sample of the dataset for testing purposes.
        
        Parameters:
        None.
        
        Returns:
        (pd.DataFrame) - the sample dataset
    """
    df = pd.read_csv(config.ASNM_CDX, sep=";")
    df.sample(n=500).to_csv(config.sample_ASNM_CDX, index=False, sep=";")
    print("===> ASNM-CDX-2009.csv")
    print("     Shape => ", df.shape)
    
    df = pd.read_csv(config.ASNM_TUN, sep=";")
    df.sample(n=200).to_csv(config.sample_ASNM_TUN, index=False, sep=";")
    print("===> ASNM-TUN.csv")
    print("     Shape => ", df.shape)
    
    df = pd.read_csv(config.ASNM_NBPO, sep=";")
    df.sample(n=500).to_csv(config.sample_ASNM_NBPO, index=False, sep=";")
    print("===> ASNM-NBPOv2.csv")
    print("     Shape => ", df.shape)
    