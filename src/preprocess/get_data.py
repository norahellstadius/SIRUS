import numpy as np 
import pandas as pd


def get_BW_data(data_path):
    full_data = pd.read_csv(data_path)
    y = full_data["diagnosis"].replace({'B': 0, 'M': 1})
    X = full_data.drop(["id", "diagnosis", "Unnamed: 32"], axis = 1)
    
    return X.to_numpy(), y
