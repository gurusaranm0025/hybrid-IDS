import warnings

from .config import ConfigClassFactory, Config
from .ingest_data import DataIngestorFactory
from .preprocessing import DataPreprocessor

from .modules.DataSplitter import DataSplitter
from .modules.DNN import DNNModel
from .modules.AE import AEDetector, AE_MODELS
from .modules.MLM import MLModel

warnings.filterwarnings('ignore')
warnings.filterwarnings('default')

def training(config: Config):    
    print(f"\n ====> {config.DATASET_NAME}:")

    print("\n ===> Loading Dataset:")
    ingestor = DataIngestorFactory.GetDataIngestor(config.DATASET_PATH)
    
    df = ingestor.ingest(config.DATASET_PATH, config.COLUMNS_TO_DROP)
    
    preprossor = DataPreprocessor(df, target_col=config.TARGET)
    preprossor.IPAddressEncoder(config.IP_COLUMNS)
    preprossor.MACAddressEncoder(config.MAC_COLUMNS)
    preprossor.target_col_val_modify(config.TARGET, config.TARGET_VAL_CHANGE)
    df = preprossor.getDataFrame()
    
    dataset = DataSplitter(df, target_col=config.TARGET, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    X_train_res,  y_train_res = dataset.get_train_resampled()
    X_train,  y_train = dataset.get_train()
    X_test, y_test = dataset.get_test()
    
    print("\n ===> DNN Model:")
    
    for dnn_mode, dnn_model_path in config.DNN_DICT.items():
        if dnn_mode == "dense":
            continue
        print(f"\n\t\t ===> DNN TRAINING FOR {dnn_mode} ===>")
        dnn_model = DNNModel(config, model_type=dnn_mode)
        dnn_model.train(X_train_res, y_train_res, X_train_t=X_train, y_train_t=y_train)
        dnn_model.model_summary()
        dnn_model.test(X_test, y_test)
        dnn_model.save_activation_values(X=dataset.X, y=dataset.y)
        dnn_model.save_model()
        dnn_model = None
        
        print(f"\n ===> AE Detector Model FOR ====> {dnn_mode}:")
        ae_detector = AEDetector(config, model_type=dnn_mode ,estimator=AE_MODELS['SVC'][0], param_grid=AE_MODELS['SVC'][1])
        ae_detector.train()
        ae_detector.test()
        ae_detector.save_model()
        ae_detector = None
    
    print("\n ===> Semi-Supervised ML Model:")
    model = MLModel(config)
    model.train(X_train, y_train)
    model.test(X_test, y_test)
    model.save_model()

if __name__ == "__main__":
    # training(ConfigClassFactory.GetConfig('CDX'))
    # training(ConfigClassFactory.GetConfig('TUN'))
    training(ConfigClassFactory.GetConfig('NBPO'))