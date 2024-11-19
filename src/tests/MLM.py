
from ..modules.DataSplitter import DataSplitter
from ..modules.MLM import MLModel, SS_MLM_PARAMS
from ..config import Config, ConfigClassFactory
from ..ingest_data import DataIngestorFactory
from ..preprocessing import DataPreprocessor

def test_MLM(config: Config):
    print(f"\n For {config.DATASET_NAME} ==>")
    ingestor = DataIngestorFactory.GetDataIngestor(config.DATASET_PATH)
    df = ingestor.ingest(config.DATASET_PATH, config.COLUMNS_TO_DROP)
    
    preprocessor = DataPreprocessor(df, config.TARGET)
    preprocessor.IPAddressEncoder(config.IP_COLUMNS)
    preprocessor.MACAddressEncoder(config.MAC_COLUMNS)
    preprocessor.target_col_val_modify(config.TARGET, config.TARGET_VAL_CHANGE)
    df = preprocessor.getDataFrame()
    
    dataset = DataSplitter(df, config.TARGET)
    
    X_train, X_test, y_train, y_test = dataset.get_train_test()
    
    model = MLModel(config=config)
    model.train(X_train, y_train)
    model.test(X_test, y_test)

if __name__ == "__main__":
    test_MLM(ConfigClassFactory.GetConfig('CDX'))
    test_MLM(ConfigClassFactory.GetConfig('TUN'))
    test_MLM(ConfigClassFactory.GetConfig('NBPO'))