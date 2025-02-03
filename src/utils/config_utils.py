import yaml
import os
import time 
from types import SimpleNamespace

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def path_name(args,it = 0):
    time_stamp = time.strftime("%dD%HH%MM%SS", time.localtime())
    embed_name = args['model']['embedding']
    back_name = args['model']['backbone']
    head_name = args['model']['task_head']
    name = f'{embed_name}_{back_name}_{head_name}_{time_stamp}_it_{it}'
    print(f'Running experiment: {name}')
    task = args['dataset']['task']
    model = args['model']['name']
    path = 'save/log/' + f'task_{task}/'+f'model_{model}/' + name
    makedir(path)
    return path

def transfer_namespace(args):
    return SimpleNamespace(**args)

if __name__ == '__main__':
    config_path = 'configs/basic.yaml'
    args = load_config(config_path)
    path_name(args)