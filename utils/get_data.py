import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
from tqdm.notebook import tqdm
warnings.filterwarnings("ignore")

def get_dataset_abspath():
    abs_script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(abs_script_path)
    parent_dir = os.path.dirname(script_dir)
    DATA_PATH = parent_dir + '/dataset/'
    return DATA_PATH
  

def get_dataset():
    DATA_PATH = get_dataset_abspath()
    datasets = [('A', 'training_setA'), ('B', 'training_setB')]
    
    data_frames = []
    patient_id_map = {}
    patient_counter = 1  

    for label, folder in datasets:
        folder_path = os.path.join(DATA_PATH, folder)
        file_list = os.listdir(folder_path)
        for file_name in tqdm(file_list, desc=f"Processing {folder}", leave=True):
            file_path = os.path.join(folder_path, file_name)
            df_temp = pd.read_csv(file_path, sep='|')
            df_temp['patient_id'] = patient_counter
            df_temp['dataset'] = label
            
            data_frames.append(df_temp)
            patient_id_map[patient_counter] = file_name
            patient_counter += 1

    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df, patient_id_map


if __name__ == '__main__':
    print("Geting Data...")