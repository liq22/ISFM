# python main.py --config_dir configs/HSE_paper/com_patch.yaml --note patch
# python main.py --config_dir configs/HSE_paper/com_positional_token.yaml --note positional_token
# python main.py --config_dir configs/HSE_paper/com_positional.yaml --note positional
# python main.py --config_dir configs/HSE_paper/com_stemb.yaml --note stemb
# python main.py --config_dir configs/HSE_paper/com_token.yaml --note token
# python main.py --config_dir configs/HSE_paper/abalation/emb_patch.yaml --note patch
# python main.py --config_dir configs/HSE_paper/abalation/emb_positional_token.yaml --note positional_token
# python main.py --config_dir configs/HSE_paper/abalation/emb_positional.yaml --note positional
# python main.py --config_dir configs/HSE_paper/abalation/emb_spatio_temporal.yaml --note stemb
# python main.py --config_dir configs/HSE_paper/abalation/emb_token.yaml --note token

CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/HSE_paper/fits/ab_fits.yaml --note fits

CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/HSE_paper/fits/ab_dlinear_patch.yaml --note ab_dlinear_patch

CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/HSE_paper/fits/ab_dlinear.yaml --note ab_dlinear

CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/HSE_paper/multi_data/improve_performance.yaml --note imporve_performance

CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/HSE_paper/fits/ab_dlinear_token.yaml --note ab_dlinear_token

CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/HSE_paper/fits/ab_manba.yaml --note ab_manba


CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/HSE_paper/fits/ab_manba_patch.yaml --note ab_manba_patch

# single

CUDA_VISIBLE_DEVICES=7 python main.py --config_dir configs/HSE_paper/fits/ab_dlinear_patch_THU.yaml --note ab_dlinear_patch_THU

CUDA_VISIBLE_DEVICES=6 python main.py --config_dir configs/HSE_paper/fits/ab_dlinear_patch_SUDA.yaml --note ab_dlinear_patch_SUDA

CUDA_VISIBLE_DEVICES=5 python main.py --config_dir configs/HSE_paper/fits/ab_dlinear_patch_CWRU.yaml --note ab_dlinear_patch_CWRU

CUDA_VISIBLE_DEVICES=4 python main.py --config_dir configs/HSE_paper/fits/ab_dlinear_token_THU.yaml --note ab_dlinear_token_THU

CUDA_VISIBLE_DEVICES=3 python main.py --config_dir configs/HSE_paper/fits/ab_dlinear_token_SUDA.yaml --note ab_dlinear_token_SUDA

CUDA_VISIBLE_DEVICES=2 python main.py --config_dir configs/HSE_paper/fits/ab_dlinear_token_CWRU.yaml --note ab_dlinear_token_CWRU