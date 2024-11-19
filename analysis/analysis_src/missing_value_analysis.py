from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MissingValueAnalsysisTempl(ABC):
    def analyze(self, df: pd.DataFrame) -> None:
        """
            Performs complete missing value analysis on the given dataset.
            
            Parameters:
            df (pandas.DataFrame) - The dataframe to analyze.
            
            Returns:
            None.
        """
        null_counts = df.isnull().sum()
        null_columns = null_counts[null_counts>0]

        if len(null_columns)>0:
            self.identify_missing_values(df)
            self.visualise_missing_value(df)
        else:
            print("No missing values.")
    
    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame) -> None:
        """
            identifies the missing value in the dataframe.
            
            Parameters:
            df (pandas.DataFrame) - The dataframe to analyze.
            
            Returns:
            None.
        """
        pass
    
    @abstractmethod
    def visualise_missing_value(self, df: pd.DataFrame) -> None:
        """
            Visualises the missing values present in the dataframe.

            Parameters:
            df (pandas.DataFrame) - The dataframe to analyze.
            
            Returns:
            None.            
        """
        pass

class SimpleMissingValueAnalysis(MissingValueAnalsysisTempl):
    
    def identify_missing_values(self, df: pd.DataFrame) -> None:
        """
            Prints the count missing in each column of the dataset.
            
            Parameters:
            df (pd.Dataframe) - The dataframe to check.
            
            Returns:
            None.
        """
        print("\nMissing Values count by columns:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values>0])
    
    def visualise_missing_value(self, df: pd.DataFrame) -> None:
        """
            Creates a heatmap to visualise the missing values in the dataframe.
            
            Parameters:
            df (pandas.DataFrame) - The dataframe to visualise.
            
            Returns:
            None.
        """
        print("\nVisualising the missing values:")
        plt.figure(figsize=(12,8))
        sns.heatmap(df.isnull(), cbar=False, cmap="virdis")
        plt.title("Missing values heatmap.")
        plt.show()