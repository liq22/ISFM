

 # 创建新会话
# tmux new -s act_linear
# conda activate LQ1
echo "开始运行 multi_data 目录下的 ab_ 消融实验..."

# # 激活函数相关实验
# echo "实验1: 激活函数 - LeakyReLU"
# CUDA_VISIBLE_DEVICES=5 python main.py --config_dir configs/HSE_paper/multi_data/ab_act_leaky_relu.yaml --note act_leaky_relu

# echo "实验2: 激活函数 - ReLU"
# CUDA_VISIBLE_DEVICES=5 python main.py --config_dir configs/HSE_paper/multi_data/ab_act_relu.yaml --note act_relu

# echo "实验3: 激活函数 - Sigmoid"
# CUDA_VISIBLE_DEVICES=5 python main.py --config_dir configs/HSE_paper/multi_data/ab_act_sigmoid.yaml --note act_sigmoid

echo "实验4: 激活函数 - Tanh"
CUDA_VISIBLE_DEVICES=5 python main.py --config_dir configs/HSE_paper/multi_data/ab_act_tanh.yaml --note act_tanh

# 网络结构相关实验
echo "实验5: 深度线性网络"
CUDA_VISIBLE_DEVICES=5 python main.py --config_dir configs/HSE_paper/multi_data/ab_deep_linear.yaml --note deep_linear

echo "实验6: 无Mixing"
CUDA_VISIBLE_DEVICES=5 python main.py --config_dir configs/HSE_paper/multi_data/ab_nomixing.yaml --note no_mixing


# 终止会话(在tmux内)
# exit