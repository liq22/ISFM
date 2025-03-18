import numpy as np
import torch
from torch.utils.data import Dataset
from .datasets import Default_dataset,SignalDataset,get_dataloaders,MultipleDataset,split_dataset
from torch.utils.data import DataLoader

DATASET_TASK_CLASS = {
    'a_006_THU': Default_dataset,
    'a_018_THU': Default_dataset,
    'a_020_DIRG': Default_dataset,
    'a_031_HUST': Default_dataset,
    'a_010_SEU': Default_dataset,
    'a_017_ottawa': Default_dataset,
    'a_000_simulation':SignalDataset,
}

DATASET_MULTI_TASK_CLASS = {
    'a_006_THU': MultipleDataset,
    'a_018_THU': MultipleDataset,
    'a_020_DIRG': MultipleDataset,
    'a_031_HUST': MultipleDataset,
    'a_010_SEU': MultipleDataset,
    'a_017_ottawa': MultipleDataset,
    'a_000_simulation':MultipleDataset,
}


def get_data(args):
    dataset_class = DATASET_TASK_CLASS[args.task]
    if args.task == 'a_000_simulation':
        dataset = SignalDataset(num_samples_per_class=300,
                                sampling_rate=1024,
                                duration=1,
                                npy_path='src/data_process/data_1024.npz',
                                regenerate=False)
        return get_dataloaders(dataset, batch_size=32)
    dataset = dataset_class(args,flag = 'train')
    train_loader = DataLoader(
        dataset = dataset,
        batch_size= args.batch_size,
        shuffle = True,
        num_workers = args.num_workers
    )
    dataset = dataset_class(args,flag = 'val')
    val_loader = DataLoader(
        dataset = dataset,
        batch_size= args.batch_size,
        shuffle = False,
        num_workers = args.num_workers
    )
    dataset = dataset_class(args,flag = 'test')
    test_loader = DataLoader(
        dataset = dataset,
        batch_size= args.batch_size,
        shuffle = False,
        num_workers = args.num_workers
    )     
    return train_loader,val_loader,test_loader

def get_multiple_data(args):
    dataset_class = DATASET_TASK_CLASS[args.task]
    
    # 定义加载器字典
    train_loaders_dict = {}
    val_loaders_dict = {}
    test_loaders_dict = {}
    
    for data_name, dataset_dict in args.task.items():
        print('### loading data:', data_name)
        source_files = dataset_dict['source'] if isinstance(dataset_dict['source'], list) else [dataset_dict['source']]
        target_files = dataset_dict['target'] if isinstance(dataset_dict['target'], list) else [dataset_dict['target']]
        data_dir = dataset_dict['data_dir']
        
        source_files_labels = [data_dir + f"{source_file}_label.npy" for source_file in source_files]
        target_files_labels = [data_dir + f"{target_file}_label.npy" for target_file in target_files]
        
        source_files_data = [data_dir + f"{source_file}_data.npy" for source_file in source_files]
        target_files_data = [data_dir + f"{target_file}_data.npy" for target_file in target_files]

        multiple_dataset_class = DATASET_MULTI_TASK_CLASS[data_name]
        
        source_dataset = multiple_dataset_class(source_files_data, source_files_labels)
        target_dataset = multiple_dataset_class(target_files_data, target_files_labels)
        
        train_dataset,val_dataset = split_dataset(source_dataset, args.train_val_rate)

        train_loader = DataLoader(
            dataset = train_dataset,
            batch_size= args.batch_size,
            shuffle = True,
            num_workers = args.num_workers
        )
        val_loader = DataLoader(
            dataset = val_dataset,
            batch_size= args.batch_size,
            shuffle = False,
            num_workers = args.num_workers
        )
        test_loader = DataLoader(
            dataset = target_dataset,
            batch_size= args.batch_size,
            shuffle = False,
            num_workers = args.num_workers
        )
        train_loaders_dict[data_name] = train_loader
        val_loaders_dict[data_name] = val_loader
        test_loaders_dict[data_name] = test_loader
    return train_loaders_dict,val_loaders_dict,test_loaders_dict
