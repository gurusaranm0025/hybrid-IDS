from abc import ABC, abstractmethod
import pandas as pd

# Abstract base class for data inspection strategies.
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> None:
        """
            performs data inspection strategy.
            
            Parameters:
            df (pd.Dataframe) : The dataframe of the dataset file we are inspecting.
            
            Returns:
            None : This method prints the inpection result and returns nothing.
        """
        pass

class DataTypeInspectionStrategy(DataInspectionStrategy):
    
    def inspect(self, df: pd.DataFrame) -> None:
        """
            Inspects and prints the data types there and prints the null value count if there's any.
            
            Parameters:
            df (pd.Dataframe) : The dataframe of the dataset file we are inspecting.
            
            Returns:
            None : This method prints the inpection result and returns nothing.
        """
        print("\nData Types and Non-Null counts:")
        print(df.info())
        
        print("\nPrinting all the columns and their data type:")
        self.print_data_types(df)
        
        print("\nColumns with null values:")        
        null_counts = df.isnull().sum()
        null_columns = null_counts[null_counts>0]
        
        if len(null_columns)>0:
            for column, count in null_columns.items():
                print(f"Column: {column}, Null Count: {count}")
        else:
            print("No null values are found in any columns.")
    
    def print_data_types(self, df: pd.DataFrame) -> None:
        """
            It prints all the column name and their data types. Used if there are more columns.
            
            Parameters:
            df (pd.Dataframe) : The dataframe of the dataset file we are inspecting.
            
            Returns:
            None : This method prints the inpection result and returns nothing.
        """
        
        data_summary = pd.DataFrame({
            'DataTypes': df.dtypes,
        }).reset_index()
        data_summary.columns = ['column_name', 'dtype']

        for _, row in data_summary.iterrows():
            print(f"Column: {row['column_name']}, Data type: {row["dtype"]}")


class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    
    def inspect(self, df: pd.DataFrame) -> None:
        """
            Prints the summary statistics for numerical and categorical columns and values.
            
            Parameters:
            df (pd.Dataframe) : The dataframe of the dataset file we are inspecting.
            
            Returns:
            None : This method prints the inpection result and returns nothing.
        """
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe(include=["int64", "float64"]))
        
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["object", "bool"]))

class DataInspector:
    
    def __init__(self, Strategy: DataInspectionStrategy) -> None:
        """
            Initialises the data inspector with specific inpection.
            
            Parameters:
            Strategy (DataInspectionStrategy) : The strategy to use for the inpection.
            
            Returns:
            None
        """
        self._strategy = Strategy
        
    def set_strategy(self, Strategy: DataInspectionStrategy) -> None:
        """
            Sets a new strategy for inspection.
            
            Parameters:
            Strategy (DataInspectionStrategy) : The strategy to use for the inpection.
            
            Returns:
            None
        """
        self._strategy = Strategy
    
    def execute_inspection(self, df: pd.DataFrame) -> None:
        """
            Executes the inpect method from the given strategy.
            
            Parameters:
            df (pd.Dataframe) : Dataframe of the dataset to inpect.
            
            Returns:
            None
        """
        self._strategy.inspect(df)

if __name__ == "__main__":
    
    pass