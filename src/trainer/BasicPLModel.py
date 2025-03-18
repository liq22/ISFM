import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
import numpy as np
from typing import Dict, List, Optional, Any




class BasicPLModel(pl.LightningModule):
    """
    General PyTorch Lightning training module
    
    Features:
    - Configurable loss function and evaluation metrics
    - Supports various regularization methods
    - Automatic noise injection
    - Flexible optimizer configuration
    """
    
    def __init__(
        self,
        network: nn.Module,
        args_t: Any,
        args_m: Any,
        args_d: Any
    ):
        """
        初始化训练模块
        
        :param network: 待训练的主干网络
        :param args_t: 训练参数配置对象
        :param args_m: 模型参数配置对象
        :param args_d: 数据参数配置对象
        """
        super().__init__()
        self.network = network # placeholder
        self.args_t = args_t
        self.args_m = args_m
        self.args_d = args_d
        
        # self.loss = nn.CrossEntropyLoss()　ｒｅｍｏｖｅ
        # self.metric_val = torchmetrics.Accuracy(task="multiclass", num_classes=args_m.n_classes)
        # self.metric_train = torchmetrics.Accuracy(task="multiclass", num_classes=args_m.n_classes)
        # self.metric_test = torchmetrics.Accuracy(task="multiclass", num_classes=args_m.n_classes)
        # 初始化组件
        self.loss_fn = self._configure_loss()
        self.metrics = self._configure_metrics()    
        # 合并保存的超参数
        args_dict = {
            **vars(args_t),
            **vars(args_m),
            **vars(args_d)
        }
        self.save_hyperparameters(args_dict,
                                  ignore=['network'])

    def _configure_loss(self) -> nn.Module:
        """配置损失函数"""
        loss_mapping = {
            "CE": nn.CrossEntropyLoss(),
            "MSE": nn.MSELoss(),
            "BCE": nn.BCEWithLogitsLoss()
        }
        if self.args_t.cla_loss not in loss_mapping:
            raise ValueError(f"不支持的损失函数类型: {self.args_t.cla_loss}"
                             f"，可选类型: {list(loss_mapping.keys())}")
        return loss_mapping[self.args_t.cla_loss]
    
    def _configure_metrics(self) -> Dict[str, torchmetrics.Metric]:
        """配置评估指标集合"""
        metric_classes = {
            "acc": torchmetrics.Accuracy,
            "f1": torchmetrics.F1Score,
            "precision": torchmetrics.Precision,
            "recall": torchmetrics.Recall
        }
        metrics = {}
        for metric_name in self.args_t.metrics:
            if metric_name not in metric_classes:
                continue
                
            metrics.update({
                f"{stage}_{metric_name}": metric_classes[metric_name](
                    task="multiclass" if self.args_d.n_classes > 2 else "binary",
                    num_classes=self.args_d.n_classes
                )
                for stage in ["train", "val", "test"]
            })
        return nn.ModuleDict(metrics)
    
            
    def forward(self, x):
        return self.network(x)
    
    def _shared_step(self, batch: tuple, stage: str) -> Dict[str, torch.Tensor]:
        """通用处理步骤"""
        x, y, data_name = batch # data_name = False,task_name = False
        
        # # 噪声注入
        # if self.args_t.snr is not None:
        #     x = self._add_awgn(x, self._generate_snr())
        # TODO data_name = False,task_name = False
        y_hat = self(x,data_name)
        loss = self.loss_fn(y_hat, y.long())
        
        # 计算指标
        metrics = {
            f"{stage}_loss": loss,
            **{name: metric(y_hat, y) for name, metric in self.metrics.items() if name.startswith(stage)}
        }
        
        # 正则化处理
        if self.args_t.regularization['flag']:
            reg_dict = self._calculate_regularization()
            metrics.update(reg_dict)
            metrics["total_loss"] = loss + reg_dict['total']
        else:
            metrics["total_loss"] = loss
            
        return metrics

    def training_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """训练步骤"""
        metrics = self._shared_step(batch, "train")
        self._log_metrics(metrics, "train")
        return metrics.get("total_loss", metrics["train_loss"])

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """验证步骤"""
        metrics = self._shared_step(batch, "val")
        self._log_metrics(metrics, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """测试步骤"""
        metrics = self._shared_step(batch, "test")
        self._log_metrics(metrics, "test")

    def _log_metrics(self, metrics: Dict[str, torch.Tensor], stage: str) -> None:
        """统一日志记录"""
        self.log_dict(
            {k: v for k, v in metrics.items() if k.startswith(stage)},
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

    def _add_awgn(self, x: torch.Tensor, snr: int) -> torch.Tensor:
        """添加加性高斯白噪声"""
        signal_power = torch.mean(x ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        return x + noise

    # def _generate_snr(self) -> int:
    #     """生成随机SNR值"""
    #     return torch.randint(
    #         min(self.args_t.snr, 0),
    #         max(self.args_t.snr, 0) + 1,
    #         (1,)
    #     ).item()
        
        
    def _calculate_regularization(self) -> torch.Tensor:
        """
        计算多种正则化项的总损失。
        - 若 self.args_t.regularization['flag'] 为 False，则不计算任何正则化，返回0。
        - 若为 True，则遍历 regularization['method'] 中的每个键值对，
        对应 (正则化类型, 权重)，并计算相应正则化损失后累加。
        
        Returns:
            torch.Tensor: 累加后的正则化损失。
        """
        reg_config = self.args_t.regularization

        # 1. 如果未开启正则化，直接返回 0
        if not reg_config.get('flag', False):
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # 2. 获取所有需要梯度的参数
        params = [p for p in self.parameters() if p.requires_grad]

        # 3. 读取并遍历各种正则化方法及其权重
        method_dict = reg_config.get('method', {})
        reg_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        reg_dict = {}
        for reg_type, weight in method_dict.items():
            if weight == 0:
                continue  # 权重为0时可跳过
            # 根据 reg_type 决定计算方式
            if reg_type.lower() == 'l1':
                reg_loss += weight * sum(torch.norm(p, 1) for p in params)

            elif reg_type.lower() == 'l2':
                reg_loss += weight * sum(torch.norm(p, 2) for p in params)
            else:
                raise ValueError(f"不支持的正则化类型: {reg_type}")
            reg_dict = reg_dict.update({reg_type: reg_loss})
        reg_dict = reg_dict.update({"total": reg_loss})
        return reg_dict

    def configure_optimizers(self) -> Dict:
        """配置优化器和学习率调度"""
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.args_t.lr,
            weight_decay=self.args_t.weight_decay
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=self.args_t.patience // 2,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.args_t.monitor,
                # "interval": "epoch",
                "frequency":  self.args_t.patience // 2
            }
        }

if __name__ == '__main__':
    # 测试用例
    from dataclasses import dataclass
    @dataclass
    class TrainingConfig:
        """训练参数配置类"""
        learning_rate: float = 1e-3
        weight_decay: float = 0.0
        monitor: str = "val_loss"
        patience: int = 10
        snr: Optional[int] = None
        l1_norm: float = 0.0
        cla_loss: str = "cross_entropy"
        metrics: List[str] = ("accuracy",)
        regularization: Optional[Dict[str, Any]] = None
    @dataclass
    class ModelConfig:
        n_classes: int = 10

    @dataclass
    class DataConfig:
        batch_size: int = 32

    # 创建模拟网络
    class DummyModel(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.fc = nn.Linear(128, n_classes)
            
        def forward(self, x):
            return self.fc(x)

    # 配置参数
    train_args = TrainingConfig(
        cla_loss="cross_entropy",
        metrics=["accuracy", "f1"],
        regularization={"type": "l1", "weight": 0.01},
        snr=20,
        learning_rate=1e-4,
        patience=10
    )
    
    model_args = ModelConfig(n_classes=10)
    data_args = DataConfig(batch_size=32)
    
    # 初始化模型
    model = BasicPLModel(
        network=DummyModel(model_args.n_classes),
        args_t=train_args,
        args_m=model_args,
        args_d=data_args
    )
    
    # 模拟数据
    batch = (torch.randn(16, 128), torch.randint(0, 10, (16,)))
    
    # 测试各阶段
    print("\n测试训练步骤:")
    train_loss = model.training_step(batch, 0)
    print(f"训练损失: {train_loss.item():.4f}")
    
    print("\n测试验证步骤:")
    model.validation_step(batch, 0)
    
    print("\n测试优化器配置:")
    optim_config = model.configure_optimizers()
    print("优化器:", type(optim_config["optimizer"]).__name__)
    print("学习率调度器:", type(optim_config["lr_scheduler"]["scheduler"]).__name__)
    
    print("\n测试前向传播:")
    output = model(torch.randn(5, 128))
    print("输出形状:", output.shape)
    

