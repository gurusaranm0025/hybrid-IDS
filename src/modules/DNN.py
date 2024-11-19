import pandas as pd
import numpy as np
import warnings

from typing import Any
from keras import models, layers, regularizers, callbacks, Model

from ..config import ConfigClassFactory, Config
from ..ingest_data import DataIngestorFactory
from ..preprocessing import DataPreprocessor
from .DataSplitter import DataSplitter
from .LID import compute_lid_df

warnings.filterwarnings("ignore")
warnings.filterwarnings("default")

class DNNModel:
    def __init__(self, config: Config) -> None:
        self.config_ = config
        self.model_ = models.Sequential()
        self.history_: str | Any = ""
        
        self.activation_model_ = None
        self.X_train_: pd.DataFrame = None
        self.y_train_: pd.DataFrame = None
        
    def _init_activation_model(self):
        layer_outputs = [layer.output for layer in self.model_.layers if 'dense' in layer.name ]
        
        self.activation_model_ = models.Model(inputs=self.model_.inputs, outputs=layer_outputs)
    
    def _init_model(self, X_train: pd.DataFrame) -> None:
        print("\n Input Shape ==> ", X_train.shape[1])
        self.model_.add(layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(0.01)))
        self.model_.add(layers.BatchNormalization())
        
        self.model_.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        self.model_.add(layers.BatchNormalization())
        
        self.model_.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        self.model_.add(layers.BatchNormalization())
        
        self.model_.add(layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        
        self.model_.add(layers.Dense(2, activation='softmax'))
        
        self.model_.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])        
        
        print("===> MODEL INITIATED:")
    
    def model_summary(self) -> None:
        """
            Prints the model summary
        """
        print("\n ==> MODEL SUMMARY")
        self.model_.summary()
    
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_train_t: pd.DataFrame = None, y_train_t: pd.DataFrame = None) -> None:
        """
            Trains the DNN model from the given datasets
            
            Paramters:
            
            Returns:
            None.
        """
        self.X_train_ = X_train #.iloc[:, self.config_.SELECTED_FEATURES]
        self.y_train_ = y_train
        self.X_train_t_ = X_train_t #.iloc[:, self.config_.SELECTED_FEATURES]
        self.y_train_t_ = y_train_t
        self._init_model(self.X_train_)
        
        # early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        self.history_ = self.model_.fit(self.X_train_, self.y_train_, epochs=50, batch_size=32, validation_split=0.2, callbacks=[])
        self._init_activation_model()
    
    def predict(self, X: pd.DataFrame) -> Any | np.ndarray:
        return self.model_.predict(X) #.iloc[:, self.config_.SELECTED_FEATURES]
    
    def test(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
        """
            Tests the model based on the given test datasets and prints the result.
            
            Parameters:
            X_test (pandas.DataFrame) - testing input features.
            y_test (pandas.DataFrame) - actual outcome of the input features.
        """
        loss, accuracy = self.model_.evaluate(X_test, y_test, verbose=2) #.iloc[:, self.config_.SELECTED_FEATURES]
        
        print("\n => TEST RESULTS")
        print(f"\n TEST LOSS : {loss}")
        print(f"\n TEST ACCURACY : {(accuracy*100):.4f}")
        
    def get_activation_values(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.activation_model_ == None:
            self._init_activation_model()
            
        activations = self.activation_model_.predict(X)
        flattened_activations = [activation.reshape(activation.shape[0], -1) for activation in activations]
        combined_activations = np.concatenate(flattened_activations, axis=1)
        # print(combined_activations)

        return pd.DataFrame(combined_activations)
    
    def save_activation_values(self, X: pd.DataFrame = pd.DataFrame({}), y: pd.DataFrame = pd.DataFrame({})):
        """
            It saves the activation values for the given 
        """
        if self.activation_model_ == None:
            self._init_activation_model()

        if X.empty:
            X = self.X_train_t_
        # else:
        #     X = X.iloc[:, self.config_.SELECTED_FEATURES]
        
        if y.empty:
            y = self.y_train_t_
            
        if not X.empty:
            activations = self.activation_model_.predict(X)
            
            flattened_activations = [activation.reshape(activation.shape[0], -1) for activation in activations]
            combined_activations = np.concatenate(flattened_activations, axis=1)
            
            act_df = pd.DataFrame(combined_activations)
            act_df.to_csv(self.config_.ACT_VALUES_DATASET_PATH, index=False)
            
            lid_df = compute_lid_df(combined_activations, self.config_.LID_K)            
            lid_df['target'] = y
            lid_df.to_csv(self.config_.LID_DATASET_PATH, index=False)
            
            print(f"\n ACTIVATION VALUES ARE SAVED IN THE PATH : {self.config_.LID_DATASET_PATH}")
            
        else:
            print("\n ACTIVATION VALUES ARE NOT SAVED.")
            print("\n TRAIN THE MODEL, TO SAVE THE ACTIVATION VALUES.")
    
    def save_model(self):
        self.model_.save(self.config_.MODEL_PATH_DNN)
    
    def load_model(self):
        self.model_ = models.load_model(self.config_.MODEL_PATH_DNN)
            

if __name__ == "__main__":
    
    config: Config = ConfigClassFactory.GetConfig('NBPO')
    
    ingestor = DataIngestorFactory.GetDataIngestor(path=config.DATASET_PATH)
    
    df = ingestor.ingest(file_path=config.DATASET_PATH, columns_to_drop=config.COLUMNS_TO_DROP)
    
    preprocessor = DataPreprocessor(df, config.TARGET)
    preprocessor.IPAddressEncoder(config.IP_COLUMNS)
    preprocessor.MACAddressEncoder(config.MAC_COLUMNS)
    preprocessor.target_col_val_modify(target_col=config.TARGET, label_intrsuct=config.TARGET_VAL_CHANGE)
    df = preprocessor.getDataFrame()
    
    data_splitter = DataSplitter(df, config.TARGET, config.TEST_SIZE, config.RANDOM_STATE)
    
    X_train, X_test, y_train, y_test = data_splitter.get_train_test()
    
    dnn_model = DNNModel(config)

    print("\n ---> TRAINING MODE:")
    dnn_model.train(X_train, y_train, X_train_t=data_splitter.X_train_true, y_train_t=data_splitter.y_train_true)
    dnn_model.model_summary()
    dnn_model.test(X_test, y_test)

    print("\n ---> LOADING MODE:")
    dnn_model.load_model()
    dnn_model.model_summary()
    dnn_model.test(X_test, y_test)
    