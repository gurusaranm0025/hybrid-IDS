import warnings
import statistics
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

from .config import ConfigClassFactory, Config
from .ingest_data import DataIngestorFactory
from .preprocessing import DataPreprocessor

from .modules.DataSplitter import DataSplitter
from .modules.DNN import DNNModel
from .modules.LID import LID
from .modules.AE import AEDetector
from .modules.MLM import MLModel

warnings.filterwarnings('ignore')

def final_pred_condition(model_1, model_2, model_3):
    if model_2 == 1:
        return model_3
    else:
        return model_1

def run(config: Config):
    print(f"\n ====> {config.DATASET_NAME}")
    
    ingestor = DataIngestorFactory.GetDataIngestor(config.DATASET_PATH)
    df = ingestor.ingest(config.DATASET_PATH, config.COLUMNS_TO_DROP)
    
    preprocessor = DataPreprocessor(df, target_col=config.TARGET)
    preprocessor.IPAddressEncoder(config.IP_COLUMNS)
    preprocessor.MACAddressEncoder(config.MAC_COLUMNS)
    preprocessor.target_col_val_modify(config.TARGET, config.TARGET_VAL_CHANGE)
    df = preprocessor.getDataFrame()
    
    dataset = DataSplitter(df, config.TARGET)

    # X_train, y_train = dataset.get_train()
    y_train = None
    X, y = dataset.getXY()

    ssml_model = MLModel(config)
    ssml_model.load_model()
    print("\n\t ==>ssml loaded")
    
    for dnn_mode, dnn_model_path in config.DNN_DICT.items():
        if dnn_mode == 'conv1d' or dnn_mode == 'dense':
            continue
        # if dnn_mode == 'dense' and config.DATASET_NAME == "CDX":
        #     continue
        
        print(f"\n\t FOR DATASET: {config.DATASET_NAME}, FOR DNN MODE: {dnn_mode} ====>\n")
        # dnn_model = DNNModel(config, model_type=dnn_mode)
        # dnn_model.load_model(X_train=X)
        
        ae_detector = AEDetector(config, model_type=dnn_mode)
        ae_detector.load_model()    
        
        lid = LID(config, model_type=dnn_mode)
        ACCs: list[float] = []
        TPRs, FPRs, F1, Recall = [], [], [], []
        
        stratifiedSplitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_STATE)
        for fold, (train_indices, test_indices) in enumerate(stratifiedSplitter.split(X, y)):
            dnn_model = DNNModel(config, model_type=dnn_mode)
            dnn_model.load_model(X_train=X)
            print("\n\t==>Model loaded.")

            X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
            
            print(f"\n => FOLD {fold+1}: \n")
            
            dnn_results = dnn_model.predict(X_test)
            dnn_results = np.argmax(dnn_results, axis=1)
            dnn_results = pd.DataFrame(dnn_results, columns=['model_1'])
            print("\n\t ==>dnn results calced")

            act_vals = dnn_model.get_activation_values(X_test)
            print("\n\t==>got act values")
            
            lid_vals = lid.compute_lid_predict(act_vals)    
            print("\n\t==>lid values calced.")
            
            ae_result = pd.DataFrame(ae_detector.predict(lid_vals), columns=['model_2'])
            print("\n got ae results")
            
            
            ssml_result = pd.DataFrame(ssml_model.predict(X_test), columns=['model_3'])
            final_df: pd.DataFrame = pd.concat([dnn_results, ae_result, ssml_result], axis=1)
            # alt_df = final_df.copy()
            # alt_df = mode(alt_df, axis=1).mode.flatten()
            
            # accuracy = accuracy_score(y_true=y_test, y_pred=alt_df)
            # print(f"\n alt accuracy ==> {(accuracy * 100):.4f}")
            
            pred = final_df.apply(lambda row: final_pred_condition(row["model_1"], row["model_2"], row["model_3"]), axis=1)
            print("\n\t==>predicted")
            accuracy = accuracy_score(y_true=y_test, y_pred=pred)
            ACCs.append(accuracy)
            print(f"\n ACCURACY SCORE : {(accuracy * 100.0):.4f}")
            print("\n CLASSIFICATION REPORT : \n", classification_report(y_test, pred))
            
            cm =  confusion_matrix(y_test, pred)
            print("\n CONFUSION MATRIX : \n", cm)
            print("\n cm ravwel ==> ",cm.ravel())
            
            TN, FP, FN, TP = cm.ravel()
            print(TN, FP, FN, TP)
            tpr = 0 if TP+FP == 0 else TP/(TP+FP)
            fpr = 0 if FP+TN == 0 else FP/(FP+TN)
            precision = 0 if TP+FP == 0 else TP / (TP+FP)
            recall = 0 if TP+FN == 0 else TP / (TP + FN)
            f1 = 0 if precision+recall == 0 else 2*(precision*recall)/(precision+recall)
            
            FPRs.append(fpr)
            TPRs.append(tpr)
            F1.append(f1)
            Recall.append(recall)
            
            print(TPRs)
            print(FPRs)
            print(F1)
            print(Recall)
            
            

        print(f"\n =>FOR DATASET: {config.DATASET_NAME}, FOR DNN MODE: {dnn_mode} ====>")
        print(f"\n\t   --> TPRs Values: {TPRs}")
        print(f"\n\t   --> FPRs Values: {FPRs}")
        print(f"\n\t   --> F1 Values: {F1}")
        print(f"\n\t   --> Recall Values: {Recall}")
        print(f"\n\t   --> Mean of Accuracy: {(statistics.mean(ACCs)*100):.4f}")
        print(f"\n\t   --> Mean of TPRs: {(statistics.mean(TPRs)*100):.4f}")
        print(f"\n\t   --> Std Dev of TPRs: {(statistics.stdev(TPRs)*100):.4f}")
        print(f"\n\t   --> Mean of FPRs: {(statistics.mean(FPRs)*100):.4f}")
        print(f"\n\t   --> Std Dev of FPRs: {(statistics.stdev(FPRs)*100):.4f}")
        print(f"\n\t   --> Mean of F1: {(statistics.mean(F1)*100):.4f}")
        print(f"\n\t   --> Std Dev of F1: {(statistics.stdev(F1)*100):.4f}")
        print(f"\n\t   --> Mean of Recall: {(statistics.mean(Recall)*100):.4f}")
        print(f"\n\t   --> Srd Dev of Recall: {(statistics.stdev(Recall)*100):.4f}")
   
if __name__ == "__main__":
    run(ConfigClassFactory.GetConfig('CDX'))
    run(ConfigClassFactory.GetConfig('TUN'))
    run(ConfigClassFactory.GetConfig('NBPO'))