import joblib
import statistics
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from ..config import ConfigClassFactory, Config
from ..ingest_data import DataIngestorFactory
from ..preprocessing import DataPreprocessor
from .DataSplitter import DataSplitter

MODELS = [
    GaussianNB(var_smoothing=1e-6), 
    SVC(kernel='linear', C=1.0), 
    DecisionTreeClassifier(criterion='entropy', splitter='random', random_state=Config.RANDOM_STATE)
    ]


def train_asnm_tradi_models(config: Config):
    print(f"\n ===> FOR {config.DATASET_NAME}")
    
    ingestor = DataIngestorFactory.GetDataIngestor(config.DATASET_PATH)
    df = ingestor.ingest(config.DATASET_PATH, config.COLUMNS_TO_DROP)
    
    preprocessor = DataPreprocessor(df, config.TARGET)
    preprocessor.IPAddressEncoder(config.IP_COLUMNS)
    preprocessor.MACAddressEncoder(config.MAC_COLUMNS)
    preprocessor.target_col_val_modify(config.TARGET, config.TARGET_VAL_CHANGE)
    df = preprocessor.getDataFrame()
    
    dataset = DataSplitter(df, config.TARGET, test_size=0.7)
    X_train, y_train = dataset.get_train()
    X_test, y_test = dataset.get_test()
    
    save_paths = [config.NAIVE_BAYES_MODEL_PATH, config.SVM_MODEL_PATH, config.DECISION_TREE_MODEL_PATH]

    for idx, model in enumerate(MODELS):
        print(f"\n   ==> {save_paths[idx].split("/")[-1]}:")
        
        estimator = model
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        print(f"\n Accuracy: {(accuracy_score(y_pred=y_pred, y_true=y_test)*100):.4f}")
        joblib.dump(estimator, save_paths[idx])

def tester(estimator: GaussianNB | SVC | DecisionTreeClassifier, X: pd.DataFrame, y:pd.DataFrame, folds):
    TPRs, FPRs, Recall, F1, ACCs = [], [], [], [], []
    for fold, (_, test_indices) in enumerate(folds):
        # print(f"\n => FOLD {fold+1}: \n")
        X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
        
        y_pred = estimator.predict(X_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        # print(f"\n Accuracy ==> {(accuracy * 100):.4f}")
        
        cm =  confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()
        
        tpr = TP/(TP+FP)
        fpr = FP/(FP+TN)
        precision = TP / (TP+FP)
        recall = TP / (TP + FN)
        f1 = 2*(precision*recall)/(precision+recall)
        
        ACCs.append(accuracy)
        FPRs.append(fpr)
        TPRs.append(tpr)
        F1.append(f1)
        Recall.append(recall)
    # print(f"\n   --> TPRs Values: {TPRs}")
    # print(f"\n   --> FPRs Values: {FPRs}")
    # print(f"\n   --> F1 Values: {F1}")
    # print(f"\n   --> Recall Values: {Recall}")
    # print(f"\n   --> Accuracy Values: {ACCs}")
    print(f"\n      --> Mean of Accuracy: {(statistics.mean(ACCs)*100):.4f}")
    print(f"\n      --> Mean of TPRs: {(statistics.mean(TPRs)*100):.4f}")
    print(f"\n      --> Std Dev of TPRs: {(statistics.stdev(TPRs)*100):.4f}")
    print(f"\n      --> Mean of FPRs: {(statistics.mean(FPRs)*100):.4f}")
    print(f"\n      --> Std Dev of FPRs: {(statistics.stdev(FPRs)*100):.4f}")
    print(f"\n      --> Mean of F1: {(statistics.mean(F1)*100):.4f}")
    print(f"\n      --> Std Dev of F1: {(statistics.stdev(F1)*100):.4f}")
    print(f"\n      --> Mean of Recall: {(statistics.mean(Recall)*100):.4f}")
    print(f"\n      --> Srd Dev of Recall: {(statistics.stdev(Recall)*100):.4f}")

        

def test_asnm_tradi_models(config: Config):
    print(f"\n ===> FOR {config.DATASET_NAME}")
    
    ingestor = DataIngestorFactory.GetDataIngestor(config.DATASET_PATH)
    df = ingestor.ingest(config.DATASET_PATH, config.COLUMNS_TO_DROP)
    
    preprocessor = DataPreprocessor(df, config.TARGET)
    preprocessor.IPAddressEncoder(config.IP_COLUMNS)
    preprocessor.MACAddressEncoder(config.MAC_COLUMNS)
    preprocessor.target_col_val_modify(config.TARGET, config.TARGET_VAL_CHANGE)
    df = preprocessor.getDataFrame()
    
    dataset = DataSplitter(df, config.TARGET)
    X, y = dataset.get_resampled_dataset()
    
    gnb = joblib.load(config.NAIVE_BAYES_MODEL_PATH)
    svm = joblib.load(config.SVM_MODEL_PATH)
    dtc = joblib.load(config.DECISION_TREE_MODEL_PATH)

    stratifiedSplitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_STATE)
    # folds = stratifiedSplitter.split(X, y)
    print(f"\n   ==> {config.NAIVE_BAYES_MODEL_PATH.split("/")[-1]}:")
    tester(gnb, X, y, stratifiedSplitter.split(X, y))
    
    print(f"\n   ==> {config.SVM_MODEL_PATH.split("/")[-1]}:")
    tester(svm, X, y, stratifiedSplitter.split(X, y))
    
    print(f"\n   ==> {config.DECISION_TREE_MODEL_PATH.split("/")[-1]}:")
    tester(dtc, X, y, stratifiedSplitter.split(X, y))
    



    
if __name__ == "__main__":
    # train_asnm_tradi_models(ConfigClassFactory.GetConfig('CDX'))
    # train_asnm_tradi_models(ConfigClassFactory.GetConfig('TUN'))
    train_asnm_tradi_models(ConfigClassFactory.GetConfig('NBPO'))
    
    # test_asnm_tradi_models(ConfigClassFactory.GetConfig('CDX'))
    # test_asnm_tradi_models(ConfigClassFactory.GetConfig('TUN'))
    test_asnm_tradi_models(ConfigClassFactory.GetConfig('NBPO'))




# if __name__ == "__main__":
#     config = ConfigClassFactory.GetConfig('TUN')
    
#     ingestor = DataIngestorFactory.GetDataIngestor(config.DATASET_PATH)
#     df = ingestor.ingest(config.DATASET_PATH, config.COLUMNS_TO_DROP)
    
#     preprocess = DataPreprocessor(df, config.TARGET)
#     preprocess.IPAddressEncoder(config.IP_COLUMNS)
#     # preprocess.MACAddressEncoder(config.MAC_COLUMNS)
#     preprocess.target_col_val_modify(config.TARGET, config.TARGET_VAL_CHANGE)
#     df = preprocess.getDataFrame()
    
#     dataset = DataSplitter(df, config.TARGET)
#     X_train, X_test, y_train, y_test = dataset.get_train_test()
    
#     y_train_semi = y_train.copy()
#     num_unlabeled = int(0.9 * len(y_train_semi))
#     y_train_semi[:num_unlabeled] = -1
    
#     lp_model = LabelPropagation(kernel='rbf', n_neighbors=20)
#     lp_model.fit(X_train, y_train_semi)
#     y_pred = lp_model.predict(X_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"\n --> Accuracy: {(accuracy * 100.0):.4f}")
#     print(f"\n --> Classification Report: \n{classification_report(y_true=y_test, y_pred=y_pred)}")
#     print(f"\n --> Confusion Matrix: \n{confusion_matrix(y_pred=y_pred, y_true=y_test)}")
