import pandas as pd
import numpy as np
import os
from pathlib import Path
sep_col = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST',
             'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
             'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
             'Bilirubin_total', 'Hct', 'Hgb', 'PTT', 'WBC', 'Platelets',
             'Bilirubin_direct', 'Fibrinogen']

# Continues Health Indicators
con_col = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']

def find_project_root(marker=".gitignore"):
    """
    walk up from the current working directory until a directory containing the
    specified marker (e.g., .gitignore) is found.
    """
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError(
        f"Project root marker '{marker}' not found starting from {current}")

def feature_slide_window(patient_data, columns):
    
    window_size = 6
    features = {}
    
    for column in columns:
        series = pd.Series(patient_data[column])

        features[f'{column}_max'] = series.rolling(window=window_size, min_periods=1).max()
        features[f'{column}_min'] = series.rolling(window=window_size, min_periods=1).min()
        features[f'{column}_mean'] = series.rolling(window=window_size, min_periods=1).mean()
        features[f'{column}_median'] = series.rolling(window=window_size, min_periods=1).median()
        features[f'{column}_std'] = series.rolling(window=window_size, min_periods=1).std()
        
        # For calculating std dev of differences, use diff() then apply rolling std
        diff_std = series.diff().rolling(window=window_size, min_periods=1).std()
        features[f'{column}_diff_std'] = diff_std

    # Convert the dictionary of features into a DataFrame
    features_df = pd.DataFrame(features)
    
    return features_df

def features_score(patient_data):
    """
    Gives score assocciated with the patient data according to the scoring systems of NEWS, SOFA and qSOFA
    """
    
    scores = np.zeros((len(patient_data), 8))
    
    for ii in range(len(patient_data)):
        HR = patient_data[ii, 0]
        if HR == np.nan:
            HR_score = np.nan
        elif (HR <= 40) | (HR >= 131):
            HR_score = 3
        elif 111 <= HR <= 130:
            HR_score = 2
        elif (41 <= HR <= 50) | (91 <= HR <= 110):
            HR_score = 1
        else:
            HR_score = 0
        scores[ii, 0] = HR_score

        Temp = patient_data[ii, 2]
        if Temp == np.nan:
            Temp_score = np.nan
        elif Temp <= 35:
            Temp_score = 3
        elif Temp >= 39.1:
            Temp_score = 2
        elif (35.1 <= Temp <= 36.0) | (38.1 <= Temp <= 39.0):
            Temp_score = 1
        else:
            Temp_score = 0
        scores[ii, 1] = Temp_score

        Resp = patient_data[ii, 6]
        if Resp == np.nan:
            Resp_score = np.nan
        elif (Resp < 8) | (Resp > 25):
            Resp_score = 3
        elif 21 <= Resp <= 24:
            Resp_score = 2
        elif 9 <= Resp <= 11:
            Resp_score = 1
        else:
            Resp_score = 0
        scores[ii, 2] = Resp_score

        Creatinine = patient_data[ii, 19]
        if Creatinine == np.nan:
            Creatinine_score = np.nan
        elif Creatinine < 1.2:
            Creatinine_score = 0
        elif Creatinine < 2:
            Creatinine_score = 1
        elif Creatinine < 3.5:
            Creatinine_score = 2
        else:
            Creatinine_score = 3
        scores[ii, 3] = Creatinine_score

        MAP = patient_data[ii, 4]
        if MAP == np.nan:
            MAP_score = np.nan
        elif MAP >= 70:
            MAP_score = 0
        else:
            MAP_score = 1
        scores[ii, 4] = MAP_score

        SBP = patient_data[ii, 3]
        Resp = patient_data[ii, 6]
        if SBP + Resp == np.nan:
            qsofa = np.nan
        elif (SBP <= 100) & (Resp >= 22):
            qsofa = 1
        else:
            qsofa = 0
        scores[ii, 5] = qsofa

        Platelets = patient_data[ii, 30]
        if Platelets == np.nan:
            Platelets_score = np.nan
        elif Platelets <= 50:
            Platelets_score = 3
        elif Platelets <= 100:
            Platelets_score = 2
        elif Platelets <= 150:
            Platelets_score = 1
        else:
            Platelets_score = 0
        scores[ii, 6] = Platelets_score

        Bilirubin = patient_data[ii, 25]
        if Bilirubin == np.nan:
            Bilirubin_score = np.nan
        elif Bilirubin < 1.2:
            Bilirubin_score = 0
        elif Bilirubin < 2:
            Bilirubin_score = 1
        elif Bilirubin < 6:
            Bilirubin_score = 2
        else:
            Bilirubin_score = 3
        scores[ii, 7] = Bilirubin_score
        
    return scores

def extract_features(patient_data, columns_to_drop = []):
    # Get the column with Sepsis Label as it is not the same for each row (check documentation)
    labels = np.array(patient_data['SepsisLabel'])
    
    if columns_to_drop:
        patient_data = patient_data.drop(columns=columns_to_drop)
    
    features_A = patient_data[sep_col + con_col].values
    
    # six-hour slide window statistics of selected columns
    columns = ['HR', 'O2Sat', 'SBP', 'MAP', 'Resp']
    features_B = feature_slide_window(patient_data, columns)

    # Score features based according to NEWS, SOFA and qSOFA
    features_C = features_score(features_A)
    
    features = np.column_stack([features_A, features_B, features_C])
    
    return features, labels

def preprocess_data(input_parquet_path, output_path):
    """
    Process imputed parquet dataset and save as new parquet file
    """
    print("Loading dataset...")
    dataset = pd.read_parquet(input_parquet_path)
    
    frames_features = []
    frames_labels = []
    
    print("Processing patients...")
    total_patients = len(set(dataset.index.get_level_values(0)))
    
    for i, patient_id in enumerate(set(dataset.index.get_level_values(0))):
        # Progress update
        print(f"Processing patient {i+1}/{total_patients}", end='\r')
        
        # Get patient data
        patient_data = dataset.loc[patient_id]
        
        # Extract features and labels
        features, labels = extract_features(patient_data)
        features = pd.DataFrame(features)
        labels = pd.DataFrame(labels)
        
        frames_features.append(features)
        frames_labels.append(labels)
    
    print("\nCombining data...")
    data_features = np.array(pd.concat(frames_features))
    data_labels = (np.array(pd.concat(frames_labels)))[:, 0]
    
    # Shuffle data
    print("Shuffling data...")
    index = np.arange(len(data_labels))
    np.random.shuffle(index)
    data_features = data_features[index]
    data_labels = data_labels[index]
    
    # Convert to DataFrame and save
    print("Saving processed dataset...")
    processed_df = pd.DataFrame(data_features)
    processed_df['SepsisLabel'] = data_labels
    processed_df.to_parquet(output_path)
    
    print("Done! Dataset saved as v2_preprocessed.parquet")
    return processed_df

if __name__ == '__main__':
    # Example usage
    root = find_project_root()  
    input_path = os.path.join(root, 'dataset', 'raw_combined_data.parquet')
    output_path = os.path.join(root, 'dataset', 'v2_preprocessed.parquet')
    
    processed_df = preprocess_data(input_path, output_path)
    
    print("\nDataset Statistics:")
    print(f"Total samples: {len(processed_df)}")
    print(f"Sepsis cases: {processed_df['SepsisLabel'].sum()}")
    print(f"Non-sepsis cases: {len(processed_df) - processed_df['SepsisLabel'].sum()}")
    imbalance_ratio = processed_df['SepsisLabel'].value_counts().iloc[0] / processed_df['SepsisLabel'].value_counts().iloc[1]
    print(f"Imbalance ratio: {imbalance_ratio}")
    
