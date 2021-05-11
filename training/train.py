import argparse
import os
import joblib
import numpy as np
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Create TabularDataset using TabularDatasetFactory
# Data is located at: "https://raw.githubusercontent.com/thom/azure-ml-engineer-capstone/main/data/heart_failure_clinical_records_dataset.csv"
path = "https://raw.githubusercontent.com/thom/azure-ml-engineer-capstone/main/data/heart_failure_clinical_records_dataset.csv"
ds = TabularDatasetFactory.from_delimited_files(path=path)


def clean_data(data):
    '''
    Clean the data
    
    Parameters:
        data (TabularDataset): Dataset to be cleaned

    Returns:
        x_df(pandas.DataFrame): Feature dataset
        y_df(pandas.DataFrame): Label dataset
    '''
    pass


def main():
    '''Perform the model training'''
    # Clean data
    x, y = clean_data(ds)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=42)

    run = Run.get_context()

    pass


if __name__ == '__main__':
    main()
