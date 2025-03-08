import torch.nn as nn

class H_01_Linear_cla(nn.Module):
    def __init__(self, args):
        super(H_01_Linear_cla, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.output_dim, args.n_classes)

    def forward(self, x):
        # x: (B, T, d_model) 先对时间维度做平均池化
        x = x.mean(dim=1)  # (B, d_modelN)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits