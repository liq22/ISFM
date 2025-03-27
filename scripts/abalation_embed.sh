

 # 创建新会话
# tmux new -s embed

# tmux 


echo "开始运行 abalation 目录下的消融实验..."

# 运行 configs/HSE_paper/abalation 目录下的实验
# echo "实验1: 嵌入 - patch"
# CUDA_VISIBLE_DEVICES=4 python main.py --config_dir configs/HSE_paper/abalation/emb_patch.yaml --note emb_patch

echo "实验2: 嵌入 - positional token"
CUDA_VISIBLE_DEVICES=4 python main.py --config_dir configs/HSE_paper/abalation/emb_positional_token.yaml --note emb_pos_token

echo "实验3: 嵌入 - positional"
CUDA_VISIBLE_DEVICES=4 python main.py --config_dir configs/HSE_paper/abalation/emb_positional.yaml --note emb_pos

echo "实验4: 嵌入 - spatio temporal"
CUDA_VISIBLE_DEVICES=4 python main.py --config_dir configs/HSE_paper/abalation/emb_spatio_temporal.yaml --note emb_spatiotemporal

echo "实验5: 嵌入 - token" 
CUDA_VISIBLE_DEVICES=4 python main.py --config_dir configs/HSE_paper/abalation/emb_token.yaml --note emb_token

echo "abalation 目录下的所有消融实验已完成。"


# 终止会话(在tmux内)
# exit