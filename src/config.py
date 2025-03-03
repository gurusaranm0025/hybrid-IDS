from typing import Literal
from abc import ABC

class Config(ABC):
        DATASET_NAME: str
        COLUMNS_TO_DROP: list
        DATASET_PATH: str
        TARGET: str
        TARGET_VAL_CHANGE: list
        IP_COLUMNS: list
        MAC_COLUMNS: list
        SELECTED_FEATURES: list
        SAMPLE_DATASET_PATH: str
        
        # ACT_VALUES_DATASET_PATH: str
        # LID_DATASET_PATH: str
        
        ACT_VALS_DENSE: str
        ACT_VALS_CONV: str
        ACT_VALS_RNN: str
        
        ACT_VALS_DICT: dict
        
        LID_DENSE: str
        LID_CONV: str
        LID_RNN: str
        
        LID_DICT: dict
        
        AE_DENSE: str
        AE_CONV: str
        AE_RNN: str
        
        AE_DICT: dict
        
        # MODEL_PATH_DNN: str
        # MODEL_PATH_AE: str
        MODEL_PATH_LP: str
        
        NAIVE_BAYES_MODEL_PATH: str
        DECISION_TREE_MODEL_PATH: str
        SVM_MODEL_PATH: str
        
        DNN_DENSE: str
        DNN_CONV: str
        DNN_RNN: str
        
        LID_DATASET_TARGET: str = "target"
        RANDOM_STATE: int = 42
        TEST_SIZE: int | float = 0.3
        LID_K: int = 16
        
        DNN_DICT: dict 
        
        
class ASNM_CDX(Config):
        DATASET_NAME = "CDX"
        COLUMNS_TO_DROP = ["label_poly", "id"]
        DATASET_PATH = "./extracted_data/ASNM-CDX-2009.csv"
        TARGET = "label_2"
        TARGET_VAL_CHANGE = [[False, 0], [True, 1]]
        IP_COLUMNS = ["SrcIP", "DstIP"] 
        MAC_COLUMNS = ["SrcMAC", "DstMAC"]
        SELECTED_FEATURES = [0,1,2,3,4,5,6,7,8,13,14,15,20,21,27,34,36,65,69,77,93,164,182,183,215,223,224,225,226,227,241,243,244,371,567,568,571,771,802,803,872]
        SAMPLE_DATASET_PATH = "./sampleData/ASNM-CDX-sample.csv"
        
        # ACT_VALUES_DATASET_PATH = "./extracted_data/activation_values_"+DATASET_NAME+".csv"
        # LID_DATASET_PATH = "./extracted_data/lid_"+DATASET_NAME+".csv"
        
        ACT_VALS_DENSE = "./extracted_data/activation_values_dense_"+DATASET_NAME+".csv"
        ACT_VALS_CONV = "./extracted_data/activation_values_conv_"+DATASET_NAME+".csv"
        ACT_VALS_RNN = "./extracted_data/activation_values_rnn_"+DATASET_NAME+".csv"
        
        ACT_VALS_DICT = {
            'dense': ACT_VALS_DENSE,
            'conv1d': ACT_VALS_CONV,
            'rnn': ACT_VALS_RNN
        }
        
        LID_DENSE = "./extracted_data/lid_dense_"+DATASET_NAME+".csv"
        LID_CONV = "./extracted_data/lid_conv_"+DATASET_NAME+".csv"
        LID_RNN = "./extracted_data/lid_rnn_"+DATASET_NAME+".csv"
        
        LID_DICT = {
            'dense': LID_DENSE,
            'conv1d': LID_CONV,
            'rnn': LID_RNN,
        }

        
        AE_DENSE = "./models/ae_dense_"+DATASET_NAME+".model"
        AE_CONV = "./models/ae_conv_"+DATASET_NAME+".model"
        AE_RNN = "./models/ae_rnn_"+DATASET_NAME+".model"

        AE_DICT = {
            'dense': AE_DENSE,
            'conv1d': AE_CONV,
            'rnn': AE_RNN
        }
        
        # MODEL_PATH_DNN = "./models/dnn_"+DATASET_NAME+".h5"
        # MODEL_PATH_AE = "./models/ae_"+DATASET_NAME+".model"
        MODEL_PATH_LP = "./models/lp_"+DATASET_NAME+".model"
        
        NAIVE_BAYES_MODEL_PATH: str = "./models/naive_bayes_"+DATASET_NAME+".model"
        DECISION_TREE_MODEL_PATH: str = "./models/decision_tree_"+DATASET_NAME+".model"
        SVM_MODEL_PATH: str = "./models/svm_"+DATASET_NAME+".model"
        
        DNN_DENSE = "./models/dnn_dense_"+DATASET_NAME+".weights.h5"
        DNN_CONV = "./models/dnn_conv_"+DATASET_NAME+".weights.h5"
        DNN_RNN = "./models/dnn_rnn_"+DATASET_NAME+".weights.h5"
        
        DNN_DICT = {
            'dense': DNN_DENSE,
            'conv1d': DNN_CONV,
            'rnn': DNN_RNN
        }



class ASNM_TUN(Config):
        DATASET_NAME = "TUN"
        COLUMNS_TO_DROP = ["label_poly", "label_poly_s", "label_2", "id"]
        DATASET_PATH = "./extracted_data/ASNM-TUN.csv"
        TARGET = "label_3"
        TARGET_VAL_CHANGE = [[1, 0],[2, 1], [3, 1]]
        IP_COLUMNS = ["SrcIP", "DstIP"]
        MAC_COLUMNS = None
        SELECTED_FEATURES = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,36,37,39,52,587]
        SAMPLE_DATASET_PATH = "./sampleData/ASNM-TUN-sample.csv"
        
        # ACT_VALUES_DATASET_PATH = "./extracted_data/activation_values_"+DATASET_NAME+".csv"
        # LID_DATASET_PATH = "./extracted_data/lid_"+DATASET_NAME+".csv"

        ACT_VALS_DENSE = "./extracted_data/activation_values_dense_"+DATASET_NAME+".csv"
        ACT_VALS_CONV = "./extracted_data/activation_values_conv_"+DATASET_NAME+".csv"
        ACT_VALS_RNN = "./extracted_data/activation_values_rnn_"+DATASET_NAME+".csv"
        
        ACT_VALS_DICT = {
            'dense': ACT_VALS_DENSE,
            'conv1d': ACT_VALS_CONV,
            'rnn': ACT_VALS_RNN
        }
        
        LID_DENSE = "./extracted_data/lid_dense_"+DATASET_NAME+".csv"
        LID_CONV = "./extracted_data/lid_conv_"+DATASET_NAME+".csv"
        LID_RNN = "./extracted_data/lid_rnn_"+DATASET_NAME+".csv"

        LID_DICT = {
            'dense': LID_DENSE,
            'conv1d': LID_CONV,
            'rnn': LID_RNN,
        }

        AE_DENSE = "./models/ae_dense_"+DATASET_NAME+".model"
        AE_CONV = "./models/ae_conv_"+DATASET_NAME+".model"
        AE_RNN = "./models/ae_rnn_"+DATASET_NAME+".model"

        AE_DICT = {
            'dense': AE_DENSE,
            'conv1d': AE_CONV,
            'rnn': AE_RNN
        }
        
        # MODEL_PATH_DNN = "./models/dnn_"+DATASET_NAME+".h5"
        # MODEL_PATH_AE = "./models/ae_"+DATASET_NAME+".model"
        MODEL_PATH_LP = "./models/lp_"+DATASET_NAME+".model"

        NAIVE_BAYES_MODEL_PATH: str = "./models/naive_bayes_"+DATASET_NAME+".model"
        DECISION_TREE_MODEL_PATH: str = "./models/decision_tree_"+DATASET_NAME+".model"
        SVM_MODEL_PATH: str = "./models/svm_"+DATASET_NAME+".model"
        
        DNN_DENSE = "./models/dnn_dense_"+DATASET_NAME+".weights.h5"
        DNN_CONV = "./models/dnn_conv_"+DATASET_NAME+".weights.h5"
        DNN_RNN = "./models/dnn_rnn_"+DATASET_NAME+".weights.h5"

        DNN_DICT = {
            'dense': DNN_DENSE,
            'conv1d': DNN_CONV,
            'rnn': DNN_RNN
        }


class ASNM_NBPO(Config):
        DATASET_NAME = "NBPO"
        COLUMNS_TO_DROP = ["label_poly", "label_2", "label_poly_o", "id"]
        DATASET_PATH = "./extracted_data/ASNM-NBPOv2.csv"
        TARGET = "label"
        TARGET_VAL_CHANGE = [[1, 0], [2, 1], [3, 1]]
        IP_COLUMNS = ["srcIP", "dstIP"]
        MAC_COLUMNS = ["srcMAC", "dstMAC"]
        # SELECTED_FERATURES = [3,4,10,11,14,33,35,52,60,66,177,271,557,572,578,598,599,798,858,871]
        SELECTED_FEATURES = [0, 1, 2, 3, 4, 8, 10, 11, 12, 13, 14, 16, 20, 30, 33, 35, 52, 60, 62, 66, 115, 171, 177, 218, 268, 271, 364, 530, 557, 572, 578, 588, 598, 599, 798, 837, 850, 858, 871, 889, 896, 898]
        SAMPLE_DATASET_PATH = "./sampleData/ASNM-NBPO-sample.csv"
        
        # ACT_VALUES_DATASET_PATH = "./extracted_data/activation_values_"+DATASET_NAME+".csv"
        # LID_DATASET_PATH = "./extracted_data/lid_"+DATASET_NAME+".csv"

        ACT_VALS_DENSE = "./extracted_data/activation_values_dense_"+DATASET_NAME+".csv"
        ACT_VALS_CONV = "./extracted_data/activation_values_conv_"+DATASET_NAME+".csv"
        ACT_VALS_RNN = "./extracted_data/activation_values_rnn_"+DATASET_NAME+".csv"

        ACT_VALS_DICT = {
            'dense': ACT_VALS_DENSE,
            'conv1d': ACT_VALS_CONV,
            'rnn': ACT_VALS_RNN
        }
        
        LID_DENSE = "./extracted_data/lid_dense_"+DATASET_NAME+".csv"
        LID_CONV = "./extracted_data/lid_conv_"+DATASET_NAME+".csv"
        LID_RNN = "./extracted_data/lid_rnn_"+DATASET_NAME+".csv"

        LID_DICT = {
            'dense': LID_DENSE,
            'conv1d': LID_CONV,
            'rnn': LID_RNN,
        }

        AE_DENSE = "./models/ae_dense_"+DATASET_NAME+".model"
        AE_CONV = "./models/ae_conv_"+DATASET_NAME+".model"
        AE_RNN = "./models/ae_rnn_"+DATASET_NAME+".model"

        AE_DICT = {
            'dense': AE_DENSE,
            'conv1d': AE_CONV,
            'rnn': AE_RNN
        }
        
        # MODEL_PATH_DNN = "./models/dnn_"+DATASET_NAME+".h5"
        # MODEL_PATH_AE = "./models/ae_"+DATASET_NAME+".model"
        MODEL_PATH_LP = "./models/lp_"+DATASET_NAME+".model"

        NAIVE_BAYES_MODEL_PATH: str = "./models/naive_bayes_"+DATASET_NAME+".model"
        DECISION_TREE_MODEL_PATH: str = "./models/decision_tree_"+DATASET_NAME+".model"
        SVM_MODEL_PATH: str = "./models/svm_"+DATASET_NAME+".model"

        DNN_DENSE = "./models/dnn_dense_"+DATASET_NAME+".weights.h5"
        DNN_CONV = "./models/dnn_conv_"+DATASET_NAME+".weights.h5"
        DNN_RNN = "./models/dnn_rnn_"+DATASET_NAME+".weights.h5"

        DNN_DICT = {
            'dense': DNN_DENSE,
            'conv1d': DNN_CONV,
            'rnn': DNN_RNN
        }

class ConfigClassFactory:
    @staticmethod
    def GetConfig(dataset: Literal['CDX', 'TUN', 'NBPO'] = 'CDX') -> Config:
        if dataset == 'CDX':
            return ASNM_CDX()
        elif dataset == 'NBPO':
            return ASNM_NBPO()
        else:
            return ASNM_TUN()

class ConfigClass:
    """
        Contains the data paths and details that needs to be tuned on the run in this class
    """
    def __init__(self) -> None:
        
        # columns to drop from the dataset
        self.COLUMNS_TO_DROP_ASNM_CDX = ["label_poly", "id"]
        self.COLUMNS_TO_DROP_ASNM_TUN = ["label_poly", "label_poly_s", "label_2", "id"]
        self.COLUMNS_TO_DROP_ASNM_NBPO = ["label_poly", "label_2", "label_poly_o", "id"]
        
        # path to sample .csv files
        self.sample_ASNM_CDX = "./sampleData/ASNM-CDX-sample.csv"
        self.sample_ASNM_TUN = "./sampleData/ASNM-TUN-sample.csv"
        self.sample_ASNM_NBPO = "./sampleData/ASNM-NBPO-sample.csv"
        
        
        # path for the full .csv files.
        self.ASNM_CDX = "./extracted_data/ASNM-CDX-2009.csv"
        self.ASNM_TUN = "./extracted_data/ASNM-TUN.csv"
        self.ASNM_NBPO = "./extracted_data/ASNM-NBPOv2.csv"
        
        # target column
        self.target_ASNM_CDX = "label_2"
        self.target_ASNM_TUN = "label_3"
        self.target_ASNM_NBPO = "label"
        
        # target class change value pairs
        self.target_val_CDX = [[False, 0], [True, 1]]
        self.target_val_TUN_NBPO = [[3, 2]]
       
        
        # IP address column names.
        self.IP_COLUMNS_CDX_TUN = ["SrcIP", "DstIP"] 
        self.IP_COLUMNS_NBPO = ["srcIP", "dstIP"]
        
        # MAC address column names.
        self.MAC_COLUMNS_CDX = ["SrcMAC", "DstMAC"]
        self.MAC_COLUMNS_NBPO = ["srcMAC", "dstMAC"]
        
        self.selected_features_CDX = [ 0, 1, 3, 4,5,6,7,8,27,34,77,182,215,371,567,571,771,802,803,872]
        self.selected_features_TUN = [ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,33,36,37,52,587]
        self.selected_features_NBPO = [3,4,10,11,14,33,35,52,60,66,177,271,557,572,578,598,599,798,858,871]

config = ConfigClass()