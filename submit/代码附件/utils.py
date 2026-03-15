import os
import pandas as pd
import numpy as np


def get_day_folders(data_path):
    folders = []
    for name in os.listdir(data_path):
        full_path = os.path.join(data_path, name)
        if os.path.isdir(full_path) and name.isdigit():
            folders.append(name)
    folders.sort(key=lambda x: int(x))
    return folders

def load_day_data(data_path, day_folder):
    day_path = os.path.join(data_path, day_folder)
    data = {}
    for stock in ['A', 'B', 'C', 'D', 'E']:
        csv_path = os.path.join(day_path, f'{stock}.csv')
        if os.path.exists(csv_path):
            data[stock] = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Missing file: {csv_path}")
    return data


def clean_data(data):
    data = np.where(np.isnan(data), 0, data)
    data = np.where(np.isinf(data), 0, data)
    data = np.where(np.isinf(-data), 0, data)
    return data


def evaluate_ic(my_preds, ground_truth):
    data = np.vstack((my_preds, ground_truth))
    data = clean_data(data)
    cor = np.corrcoef(data)[0, 1]
    return cor
      
            
