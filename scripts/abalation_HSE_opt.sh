

 # 创建新会话
# tmux new -s opt
# conda activate LQ1
# 采样方式相关实验
echo "实验13: 采样方式"
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/HSE_paper/multi_data/ab_sample.yaml --note sampling

# 优化器相关实验
# echo "实验14: SGD优化器"
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/HSE_paper/multi_data/ab_sgd.yaml --note sgd


echo "实验15: SGD优化器"
CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/HSE_paper/multi_data/ab_sgd.yaml --note adamw

# 重新接入会话
tmux attach -t opt

# 终止会话(在tmux内)
# exit