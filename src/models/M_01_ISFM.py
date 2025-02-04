from .embedding import *
from .backbone import *
from .task_head import *
import torch.nn as nn

Embedding_dict = {
    'E_01_HTFE': E_01_HTFE,
}
Backbone_dict = {
    'B_01_basic_transformer': B_01_basic_transformer,
}
TaskHead_dict = {
    'H_01_Linear_cla': H_01_Linear_cla,
}
class M_01_ISFM(nn.Module):
    def __init__(self, args_m):
        super(M_01_ISFM, self).__init__()
        self.embedding = Embedding_dict[args_m.embedding](args_m)
        self.backbone = Backbone_dict[args_m.backbone](args_m)
        self.task_head = TaskHead_dict[args_m.task_head](args_m)
    def forward(self, x):
        x = self.embedding(x)
        x = self.backbone(x)
        x = self.task_head(x)
        return x
    
if __name__ == '__main__':
    
    # 在 M_01_ISFM.py 文件开头添加
    import sys
    import os

    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 计算项目根目录路径（假设文件在 src/models/ 下）
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.append(project_root)
    
    from utils.config_utils import *
    import torch
    config_path = 'configs/basic.yaml'
    args = load_config(config_path)
    args_m = transfer_namespace(args['model'])
    model = M_01_ISFM(args_m)
    print(model)
    x = torch.randn(2, 1280, 3)
    y = model(x)
    print(y.shape)