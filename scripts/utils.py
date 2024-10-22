import pandas as pd
from typing import Any
from sklearn.model_selection import train_test_split

def raw_taxi_df(filename: str) -> pd.DataFrame:
    """
    Load raw taxi dataframe from parquet
    
    Args:
        filename (string): relative path to the dataset in parquet format
    
    Returns:
        dataframe with the taxi data (pd.dataframe) 
    """
    return pd.read_parquet(path=filename)

def clean_taxi_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a clean taxi DataFrame that throws out non-numerical or outlying numerical values
    
    Args:
        raw_df (pd.DataFrame): Raw dataframe of taxi data.

    Returns:
        clean dataset.
    """
    # drop nans
    clean_df = raw_df.dropna()
    # remove trips longer than 100
    clean_df = clean_df[clean_df["trip_distance"] < 100]
    # add columns for travel time deltas and time minutes
    clean_df["time_deltas"] = clean_df["tpep_dropoff_datetime"] - clean_df["tpep_pickup_datetime"]
    clean_df["time_mins"] = pd.to_numeric(clean_df["time_deltas"]) / 6**10
    return clean_df

def split_taxi_data(clean_df: pd.DataFrame, 
                    x_column: str, 
                    y_column: str, 
                    train_size: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
        Split an x, y dataset selected from a clean dataframe; return x_train, y_train, x_test, y_test

        Args:
            clean_df (pd.DataFrame): clean dataset
            x_column: predictor variable column
            y_column: response variable column
            train_size: proportion of train samples

        Returns:
            A tuple (train data, test data) both in dataframe form   
    """
    return train_test_split(clean_df[x_column], clean_df[[y_column]], train_size=train_size)