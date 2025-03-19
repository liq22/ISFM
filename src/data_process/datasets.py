import numpy as np
import torch
from torch.utils.data import Dataset
from .simulation_signal import SignalDataset,get_dataloaders
from torch.utils.data import random_split

class Default_dataset(Dataset): # THU_006or018_basic
    def __init__(self, args,flag): # 1hz, 10hz, 15hz,IF
        self.flag = flag
        self.data_loader(args.data_dir,args.target)
        self.data_create()
        # Load data and labels 
    def data_loader(self,data_dir,target):
        
        self.data = np.load(data_dir + target + '_data.npy').astype(np.float32) # TODO remove
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
    
class MultipleDataset(Dataset):
    def __init__(self, data_files, label_files,data_name, transform=None, downsample_factor = 1):
        """
        Args:
            data_files (list): List of paths to data .npy files.
            label_files (list): List of paths to label .npy files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = []
        self.labels = []
        self.data_name = data_name
        # self.transform = transform
        # self.downsample_factor = downsample_factor

        for data_file, label_file in zip(data_files, label_files):
            # Load data and labels
            data = np.load(data_file)  # Shape: (N_samples, L)
            labels = np.load(label_file)  # Shape: (N_samples,)

            # Append to the dataset
            self.data.append(data)
            self.labels.append(labels)

        # Concatenate all data and labels
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def downsample_signal(self,signal, factor):
        """
        Downsamples the signal by the given factor.
        
        Args:
            signal (numpy array): The original signal.
            factor (int): The downsampling factor.
            
        Returns:
            numpy array: The downsampled signal.
        """
        return signal[::factor]
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]

        # If the signal is 1D, add a channel dimension
        if len(signal.shape) == 1:
            signal = np.expand_dims(signal, axis=-1)  # Shape: (L, 1)
        # if self.downsample_factor > 1:
        #     signal = self.downsample_signal(signal, self.downsample_factor)
        # if self.transform:
        #     signal = self.transform(signal)
        signal = torch.tensor(signal, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return signal, label # ,self.data_name  # fix bug
        # return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def split_dataset(dataset, train_ratio=0.7): # , val_ratio=0.1, test_ratio=0.2
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size #  - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    return train_dataset, val_dataset# , test_dataset