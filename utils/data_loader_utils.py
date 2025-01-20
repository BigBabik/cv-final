import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def load_calibration_data(base_dir):
    calibration_data = {}
    for subdir in os.listdir(base_dir):
        calib_path = os.path.join(base_dir, subdir, "calibration.csv")
        if os.path.exists(calib_path):
            df = pd.read_csv(calib_path)
            for _, row in df.iterrows():
                
                calibration_data[row['image_id']] = {
                    'K': np.array([np.float64(x) for x in row['camera_intrinsics'].split(" ")]).reshape(3, 3),
                    'R': np.array([np.float64(x) for x in row['rotation_matrix'].split(" ")]).reshape(3, 3),
                    'T': np.array([np.float64(x) for x in row['translation_vector'].split(" ")])
                }
    return calibration_data

def load_pair_data(base_dir):
    pairs = []
    for subdir in os.listdir(base_dir):
        pair_path = os.path.join(base_dir, subdir, "pair_covisibility.csv")
        if os.path.exists(pair_path):
            df = pd.read_csv(pair_path)
            for _, row in df.iterrows():
                pairs.append({
                    'pair': row['pair'],
                    'covisibility': row['covisibility'],
                    'fundamental_matrix': np.array([np.float64(x) for x in row['fundamental_matrix'].split(" ")]).reshape(3, 3)
                })
    print(f"Found {len(pairs)} pairs")
    return pairs


class FundamentalMatrixDataset(Dataset):
    def __init__(self, pairs, calibration_data, min_covisibility=0.1):
        self.pairs = [p for p in pairs if p['covisibility'] >= min_covisibility]
        self.calibration_data = calibration_data

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        key1, key2 = pair['pair'].split('-')
        K1, R1, T1 = self.calibration_data[key1]['K'], self.calibration_data[key1]['R'], self.calibration_data[key1]['T']
        K2, R2, T2 = self.calibration_data[key2]['K'], self.calibration_data[key2]['R'], self.calibration_data[key2]['T']

        F_gt = torch.tensor(pair['fundamental_matrix'], dtype=torch.float32)

        # Return calibration matrices and ground truth
        return {
            'K1': torch.tensor(K1, dtype=torch.float32),
            'R1': torch.tensor(R1, dtype=torch.float32),
            'T1': torch.tensor(T1, dtype=torch.float32),
            'K2': torch.tensor(K2, dtype=torch.float32),
            'R2': torch.tensor(R2, dtype=torch.float32),
            'T2': torch.tensor(T2, dtype=torch.float32),
            'F_gt': F_gt
        }



def create_dataset(train_data_path: str):
    # Load data
    if not os.path.exists(train_data_path):
        raise Exception(f"cant find train data path {train_data_path}")
    calibration_data = load_calibration_data(train_data_path)
    pair_data = load_pair_data(train_data_path)


    # Create dataset and data loader
    dataset = FundamentalMatrixDataset(pair_data, calibration_data)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataset ,data_loader
