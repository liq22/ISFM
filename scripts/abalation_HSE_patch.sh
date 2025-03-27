

 # 创建新会话
# tmux new -s patch
# conda activate LQ1
# Patch数量相关实验

# echo "实验7: Patch数量 x2"
CUDA_VISIBLE_DEVICES=6 python main.py --config_dir configs/HSE_paper/multi_data/ab_np_2.yaml --note np_2x

echo "实验8: Patch数量 x3"
CUDA_VISIBLE_DEVICES=6 python main.py --config_dir configs/HSE_paper/multi_data/ab_np_3.yaml --note np_3x

echo "实验9: Patch数量 x0.5"
CUDA_VISIBLE_DEVICES=6 python main.py --config_dir configs/HSE_paper/multi_data/ab_np_05.yaml --note np_0.5x

# Patch尺寸相关实验
echo "实验10: Patch尺寸 1v2"
CUDA_VISIBLE_DEVICES=6 python main.py --config_dir configs/HSE_paper/multi_data/ab_pl_1v2.yaml --note pl_1v2

echo "实验11: Patch尺寸 2v2" 
CUDA_VISIBLE_DEVICES=6 python main.py --config_dir configs/HSE_paper/multi_data/ab_pl_2v2.yaml --note pl_2v2

echo "实验12: Patch尺寸 0.5v1"
CUDA_VISIBLE_DEVICES=6 python main.py --config_dir configs/HSE_paper/multi_data/ab_pl_05v1.yaml --note pl_05v1


# 终止会话(在tmux内)
# exit