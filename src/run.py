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
    X, y = dataset.getXY()
    
    dnn_model = DNNModel(config)
    dnn_model.load_model()
    
    ae_detector = AEDetector(config)
    ae_detector.load_model()
    
    ssml_model = MLModel(config)
    ssml_model.load_model()
    
    lid = LID(config)
    ACCs: list[float] = []
    TPRs, FPRs, F1, Recall = [], [], [], []
    
    stratifiedSplitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_STATE)
    for fold, (train_indices, test_indices) in enumerate(stratifiedSplitter.split(X, y)):
        X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
        
        print(f"\n => FOLD {fold+1}: \n")
        dnn_results = dnn_model.predict(X_test)
        dnn_results = np.argmax(dnn_results, axis=1)
        dnn_results = pd.DataFrame(dnn_results, columns=['model_1'])

        act_vals = dnn_model.get_activation_values(X_test)
        lid_vals = lid.compute_lid_predict(act_vals)    
        ae_result = pd.DataFrame(ae_detector.predict(lid_vals), columns=['model_2'])
        
        ssml_result = pd.DataFrame(ssml_model.predict(X_test), columns=['model_3'])
        final_df: pd.DataFrame = pd.concat([dnn_results, ae_result, ssml_result], axis=1)
        alt_df = final_df.copy()
        alt_df = mode(alt_df, axis=1).mode.flatten()
        
        accuracy = accuracy_score(y_true=y_test, y_pred=alt_df)
        print(f"\n alt accuracy ==> {(accuracy * 100):.4f}")
        
        pred = final_df.apply(lambda row: final_pred_condition(row["model_1"], row["model_2"], row["model_3"]), axis=1)
        
        accuracy = accuracy_score(y_true=y_test, y_pred=pred)
        ACCs.append(accuracy)
        print(f"\n ACCURACY SCORE : {(accuracy * 100.0):.4f}")
        print("\n CLASSIFICATION REPORT : \n", classification_report(y_test, pred))
        
        cm =  confusion_matrix(y_test, pred)
        print("\n CONFUSION MATRIX : \n", cm)
        TN, FP, FN, TP = cm.ravel()
        
        tpr = TP/(TP+FP)
        fpr = FP/(FP+TN)
        precision = TP / (TP+FP)
        recall = TP / (TP + FN)
        f1 = 2*(precision*recall)/(precision+recall)
        
        FPRs.append(fpr)
        TPRs.append(tpr)
        F1.append(f1)
        Recall.append(recall)

    print(f"\n   --> TPRs Values: {TPRs}")
    print(f"\n   --> FPRs Values: {FPRs}")
    print(f"\n   --> F1 Values: {F1}")
    print(f"\n   --> Recall Values: {Recall}")
    print(f"\n   --> Mean of Accuracy: {(statistics.mean(ACCs)*100):.4f}")
    print(f"\n   --> Mean of TPRs: {(statistics.mean(TPRs)*100):.4f}")
    print(f"\n   --> Std Dev of TPRs: {(statistics.stdev(TPRs)*100):.4f}")
    print(f"\n   --> Mean of FPRs: {(statistics.mean(FPRs)*100):.4f}")
    print(f"\n   --> Std Dev of FPRs: {(statistics.stdev(FPRs)*100):.4f}")
    print(f"\n   --> Mean of F1: {(statistics.mean(F1)*100):.4f}")
    print(f"\n   --> Std Dev of F1: {(statistics.stdev(F1)*100):.4f}")
    print(f"\n   --> Mean of Recall: {(statistics.mean(Recall)*100):.4f}")
    print(f"\n   --> Srd Dev of Recall: {(statistics.stdev(Recall)*100):.4f}")
    
    
    
    
if __name__ == "__main__":
    run(ConfigClassFactory.GetConfig('CDX'))
    run(ConfigClassFactory.GetConfig('TUN'))
    run(ConfigClassFactory.GetConfig('NBPO'))