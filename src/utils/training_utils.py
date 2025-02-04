from pytorch_lightning.callbacks import Callback
import torch
import copy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
import torch.nn.functional as F
import numpy as np

def cosine_similarity(x, y):
    return F.cosine_similarity(x, y, dim=1)

def random_shuffle_channels(tensor):
    # 获取C通道的数量
    C = tensor.size(1)
    
    # 生成随机的C通道索引
    perm = torch.randperm(C)
    
    # 打乱C通道
    shuffled_tensor = tensor[:, perm]

    return shuffled_tensor

def wgn2(x, snr):
    "加随机噪声"
    snr = 10**(snr/10.0)
    xpower = torch.sum(x**2)/(x.size(0)*x.size(1)*x.size(2))
    npower = xpower / snr
    return torch.randn(x.size()).cuda() * torch.sqrt(npower) + x 

def l1_reg(param):
    return torch.sum(torch.abs(param))

def sim_reg(tensor):
    shuffle_tensor = random_shuffle_channels(tensor)
    sim = cosine_similarity(tensor, shuffle_tensor)
    sim_sum = l1_reg(sim)
    return sim_sum
    
def mixup(batch,alpha = 0.8):
    # mix_ratio = np.random.dirichlet(np.ones(3) * 0.9,size=1) # 设置为0.9
    lamda = np.random.beta(alpha,alpha)
    x,y = batch
    index = torch.randperm(x.size(0)).cuda()
    
    x = lamda * x + (1-lamda) * x[index]
    y = lamda * y + (1-lamda) * y[index]
        
    return x,y

def check_attr(args,attr = 'attention_norm'):
    if not hasattr(args, attr):
        setattr(args, attr, False)
        
def load_best_model_checkpoint(model: LightningModule, trainer: Trainer) -> LightningModule:
    """
    加载训练过程中保存的最佳模型检查点。

    参数:
    - model: 要加载检查点权重的模型实例。
    - trainer: 用于训练模型的训练器实例。

    返回:
    - 加载了最佳检查点权重的模型实例。
    """
    # 从trainer的callbacks中找到ModelCheckpoint实例，并获取best_model_path
    model_checkpoint = None
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            model_checkpoint = callback
            break

    if model_checkpoint is None:
        raise ValueError("ModelCheckpoint callback not found in trainer's callbacks.")

    best_model_path = model_checkpoint.best_model_path
    print(f"Best model path: {best_model_path}")

    # 确保最佳模型路径不是空的
    if not best_model_path:
        raise ValueError("No best model path found. Please check if the training process saved checkpoints.")

    # 加载最佳检查点
    state_dict = torch.load(best_model_path)
    model.load_state_dict(state_dict['state_dict'])
    return model