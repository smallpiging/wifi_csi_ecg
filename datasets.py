import os
import torch
from pyqtgraph.examples.AxisItem_label_overlap import x_data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class CSIDataset(Dataset):
    def __init__(self, data_dir, window_size=500, step=50):
        self.window_x = []
        self.window_y = []

        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(data_dir, csv_file))
            y_data = df['ECG_Heatmap_Label'].values
            # x_data = df.drop(columns=['WIFI_Timestamp','ECG_Heatmap_Label','Aligned_Breath','WIFI_Mean_Mag']).values
            x_data = df.drop(columns=['ECG_Heatmap_Label']).values
            for i in range(0, len(x_data) - window_size, step):
                self.window_x.append(x_data[i:i+window_size].T)
                self.window_y.append(np.expand_dims(y_data[i:i+window_size], axis=0))

    def __len__(self):
        return len(self.window_x)

    def __getitem__(self, idx):
        x_tensor = torch.tensor(self.window_x[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.window_y[idx], dtype=torch.float32)
        return x_tensor, y_tensor

if __name__ =='__main__':
    # dataset = CSIDataset(data_dir='./processed_datasets', window_size=200, step=20)
    dataset = CSIDataset(data_dir='./pca_datasets', window_size=200, step=20)
    train_dataset = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    for batch_x, batch_y in train_dataset:
        print(batch_x.shape, batch_y.shape)