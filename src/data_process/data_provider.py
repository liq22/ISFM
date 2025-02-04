import numpy as np
import torch
from torch.utils.data import Dataset
from .datasets import Default_dataset
from torch.utils.data import DataLoader

DATASET_TASK_CLASS = {
    'a_006_THU': Default_dataset,
    'a_018_THU': Default_dataset,
    'a_020_DIRG': Default_dataset,
    'a_031_HUST': Default_dataset,
    'a_010_SEU': Default_dataset,
    'a_017_ottawa': Default_dataset,
}


def get_data(args):
    dataset_class = DATASET_TASK_CLASS[args.task]
    
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