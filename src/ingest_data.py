import os
import pandas as pd
from abc import ABC, abstractmethod

# Abstract class for data ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str, columns_to_drop: list, sep: str) -> pd.DataFrame:
        """abstrct method to ingest data from a given file"""
        pass
    
class CSVDataIngestor(DataIngestor):
     def ingest(self, file_path: str, columns_to_drop: list = [], sep: str = ";") -> pd.DataFrame:
         if not file_path.endswith(".csv"):
             raise ValueError("Provided file is not a .csv file.")
         
         df = pd.read_csv(file_path, sep=sep)
         df.columns = df.columns.str.strip()
        #  print(df.columns)
         
         return df.drop(columns=columns_to_drop)
    
class DataIngestorFactory:
    
    @staticmethod
    def GetDataIngestor(path: str) -> DataIngestor:
        """
            Gives the respective Data Ingestor based on the file type or extension
            
            Parameters:
            path (string) - path to the dataset.
            
            Returns:
            (DataIngestor) - a data ingestor class to ingest the data.
        """
        file_extension = os.path.splitext(path)[1]
        
        if file_extension == ".csv":
            return CSVDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")

if __name__ == "__main__":
    print("passing the script, if u want to run this file separately, then come and uncomment or modufy the lines below.")
    pass
    # file_path = "/home/saran/Projects/Clg/Final Year/Exp./Datasets/extracted_datasets/archive.zip"
    
    # file_extension = os.path.splitext(file_path)[1]
    
    # print("File path ==> ", file_path)
    # print("File extension ==> ", file_extension)
    
    # data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    
    # df = data_ingestor.ingest(file_path)
    
    # print(df.head())
    # print("Dataset shape ==> ", df.shape)
    # print("Dataset size ==> ", df.size)