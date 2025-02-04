import numpy as np
import torch
from torch.utils.data import Dataset

class Default_dataset(Dataset): # THU_006or018_basic
    def __init__(self, args,flag): # 1hz, 10hz, 15hz,IF
        self.flag = flag
        self.data_loader(args.data_dir,args.target)
        self.data_create()
        # Load data and labels 
    def data_loader(self,data_dir,target):
        
        self.data = np.load(data_dir + target + '_data.npy').astype(np.float32)
        self.labels = np.load(data_dir + target + '_label.npy').astype(np.float32)
        
    def data_create(self):
       #  Define split ratios
        train_ratio = 0.6
        val_ratio = 0.1
        # Calculate test_ratio to ensure ratios sum to 1
        test_ratio = 1 - (train_ratio + val_ratio)

        # Split indices for each label
        train_indices, val_indices, test_indices = [], [], []
        for label in np.unique(self.labels):
            label_indices = np.where(self.labels == label)[0]

            
            n_train = int(len(label_indices) * train_ratio)
            n_val = int(len(label_indices) * val_ratio)

            n_test = len(label_indices) - n_train - n_val

            # Append indices for each set
            train_indices.extend(label_indices[:n_train])
            val_indices.extend(label_indices[n_train:n_train + n_val])
            test_indices.extend(label_indices[n_train + n_val:])

        # Select indices based on the flag
        if self.flag == 'train':
            selected_indices = train_indices
        elif self.flag == 'val':
            selected_indices = val_indices
        elif self.flag == 'test':
            selected_indices = test_indices
        else:
            raise ValueError("Invalid flag. Please choose from 'train', 'val', or 'test'.")

        self.selected_data = self.data[selected_indices]
        self.selected_labels = self.labels[selected_indices]

    def __len__(self):
        return len(self.selected_data)

    def __getitem__(self, idx):
        sample = self.selected_data[idx]
        label = self.selected_labels[idx]
        
        return sample, label