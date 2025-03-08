
import argparse
import os
import pandas as pd
import wandb

from src.utils.config_utils import load_config, makedir, path_name, transfer_namespace
from src.utils.training_utils import load_best_model_checkpoint

import torch
from pytorch_lightning import seed_everything
from src.trainer.Basic_trainer import trainer_set
from src.trainer.BasicPLModel import BasicPLModel

from src.models.M_01_ISFM import M_01_ISFM

MODEL_DICT = {
    'M_01_ISFM': M_01_ISFM,
}

if __name__ == '__main__':
    iteration = 1
    # -----------------------
    # 1. 解析命令行参数
    # -----------------------
    parser = argparse.ArgumentParser(description='ISFM Runner')

    # 添加参数
    parser.add_argument('--config_dir', type=str, default='configs/basic.yaml',
                        help='The directory of the configuration file')
    meta_args = parser.parse_args()
    # -----------------------
    # 2. 加载配置文件
    # -----------------------
    config_dir = meta_args.config_dir
    configs = load_config(config_dir)
    args_t,args_m,args_d = (
    transfer_namespace(configs['trainer']),
    transfer_namespace(configs['model']),
    transfer_namespace(configs['dataset']) 
    )
    
    # -----------------------
    # 3. 多次迭代训练与测试
    # -----------------------
    for it in range(iteration):
        path, name = path_name(configs,it)
        seed_everything(args_t.seed + it) # 17 args.seed 
        # 如果需要使用 WandB，就进行初始化；否则跳过
        if args_t.wandb:
            wandb.init(project=args_t.task, name=name, notes=args_t.notes)
        else:
            wandb.init(mode='disabled')  # 避免 wandb 报错，可使用 "disabled" 模式
        # -----------------------
        # 3.1 初始化模型
        # -----------------------
        # 根据配置文件获取模型类并实例化
        model_plain = MODEL_DICT[args_m.name](args_m)
        model = BasicPLModel(model_plain, args_t,args_m,args_d)
        print(model.network)
        
        trainer,train_dataloader, val_dataloader, test_dataloader = trainer_set(args_t,args_d,path)

        # train
        trainer.fit(model,train_dataloader, val_dataloader) # TODO load best checkpoint
        model = load_best_model_checkpoint(model,trainer)
        result = trainer.test(model,test_dataloader)
        # 保存结果
        result_df = pd.DataFrame(result)
        result_df.to_csv(os.path.join(path, f'test_result_{it}.csv'), index=False)
        if args_t.wandb:
            wandb.finish()
