
from ..modules.AE import AEDetector, AE_MODELS
# from ..modules.DataSplitter import DataSplitter
from ..config import Config, ConfigClassFactory
# from ..ingest_data import DataIngestorFactory


def test_ae(config: Config):
    print("\n Dataset Name ===> ", config.DATASET_NAME)
    
    ae_detector = AEDetector(config=config)
    
    for name, model_params in AE_MODELS.items():
        print("\n Model ==> ", name)
        
        ae_detector.set_model(estimator=model_params[0], param_grid=model_params[1])
        ae_detector.train()
        ae_detector.test()

if __name__ == "__main__":
    test_ae(ConfigClassFactory.GetConfig('TUN'))