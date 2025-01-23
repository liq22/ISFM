# Industrial signal foundation model


ProjectRoot
├── README.md                       # 项目说明及使用指南
├── .gitignore                      # 忽略文件/文件夹配置
│
├── configs                         # 所有项目配置，按需细分
│   ├── basic.yaml                  # 示例基本配置文件
│   └── ...                         # (可拓展更多config，如dev.yaml, prod.yaml等)
│
├── scripts                         # 存放脚本（数据预处理、模型可视化、测试驱动等）
│   ├── preprocess_data.py          # 预处理脚本
│   ├── postprocess_data.py         # 后处理脚本
│   ├── train.py                    # 主要训练脚本/启动入口
│   ├── evaluate.py                 # 评估/推理脚本
│   └── ...                         # (可拓展更多脚本)
│
├── src                             # 核心源码
│   ├── __init__.py
│   ├── data_process                # 数据加载与处理
│   │   └── ...
│   ├── losses                      # 损失函数 & 评估指标
│   │   ├── eval_metric
│   │   │   └── ...
│   │   └── task_loss
│   │       └── ...
│   ├── models                      # 模型相关
│   │   ├── backbone.py             # Backbone骨干网络
│   │   ├── embedding.py            # Embedding等输入表示层
│   │   ├── task_head.py            # 任务头，如分类、回归等
│   │   └── z_model_collection      # 不同模型变体的集合
│   │       └── ...
│   ├── trainer                     # 训练逻辑封装 (PyTorch Lightning 模块/训练器)
│   │   ├── __init__.py
│   │   └── lightning_trainer.py    # 自定义LightningModule/LightningDataModule等
│   └── utils                       # 工具函数，日志、配置解析等
│       └── ...
│
├── test                            # 单元测试 & 集成测试
│   └── ...
│
├── save                            # 训练或测试后的输出保存路径
│   ├── log                         # 默认日志/检查点保存路径
│   └── plot                        # 可视化结果或图表
│
└── requirements.txt (或 pyproject.toml / setup.py)   # 依赖包或环境管理
