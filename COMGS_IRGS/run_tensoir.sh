# armadillo
CUDA_VISIBLE_DEVICES=4 python train_refgaussian.py -s data/TensoIR_Synthetic/armadillo -m outputs/TensoIR_Synthetic/armadillo/refgs --eval -w --lambda_mask_entropy 0.05

CUDA_VISIBLE_DEVICES=4 python train.py -s data/TensoIR_Synthetic/armadillo --eval -m outputs/TensoIR_Synthetic/armadillo/irgs  --iterations 20000 --start_checkpoint_refgs outputs/TensoIR_Synthetic/armadillo/refgs/chkpnt50000.pth --envmap_resolution 128 --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 --diffuse_sample_num 256 --envmap_cubemap_lr 0.01 --lambda_light_smooth 0.0005 --init_roughness_value 0.6 --lambda_light 0.1 --train_ray

CUDA_VISIBLE_DEVICES=4 python render.py -m outputs/TensoIR_Synthetic/armadillo/irgs --eval --diffuse_sample_num 512 --no_save --no_lpips
CUDA_VISIBLE_DEVICES=4 python compute_albedo_scale_tensoir.py -m outputs/TensoIR_Synthetic/armadillo/irgs
CUDA_VISIBLE_DEVICES=4 python eval_material_tensoir.py -m outputs/TensoIR_Synthetic/armadillo/irgs --no_save --no_lpips --albedo_rescale 2
CUDA_VISIBLE_DEVICES=4 python eval_relighting_tensoir.py -m outputs/TensoIR_Synthetic/armadillo/irgs --diffuse_sample_num 512 --light_sample_num 256 --albedo_rescale 2 -e light   

# ficus
CUDA_VISIBLE_DEVICES=5 python train_refgaussian.py -s data/TensoIR_Synthetic/ficus -m outputs/TensoIR_Synthetic/ficus/refgs --eval -w --lambda_mask_entropy 0.05

CUDA_VISIBLE_DEVICES=5 python train.py -s data/TensoIR_Synthetic/ficus --eval -m outputs/TensoIR_Synthetic/ficus/irgs  --iterations 20000 --start_checkpoint_refgs outputs/TensoIR_Synthetic/ficus/refgs/chkpnt50000.pth --envmap_resolution 128 --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 --diffuse_sample_num 256 --envmap_cubemap_lr 0.01 --lambda_light_smooth 0.0005 --init_roughness_value 0.6 --lambda_light 0.1 --train_ray

CUDA_VISIBLE_DEVICES=5 python render.py -m outputs/TensoIR_Synthetic/ficus/irgs --eval --diffuse_sample_num 512 --no_save --no_lpips
CUDA_VISIBLE_DEVICES=5 python compute_albedo_scale_tensoir.py -m outputs/TensoIR_Synthetic/ficus/irgs
CUDA_VISIBLE_DEVICES=5 python eval_material_tensoir.py -m outputs/TensoIR_Synthetic/ficus/irgs --no_save --no_lpips --albedo_rescale 2
CUDA_VISIBLE_DEVICES=5 python eval_relighting_tensoir.py -m outputs/TensoIR_Synthetic/ficus/irgs --diffuse_sample_num 512 --light_sample_num 256 --albedo_rescale 2 -e light   

# hotdog
CUDA_VISIBLE_DEVICES=6 python train_refgaussian.py -s data/TensoIR_Synthetic/hotdog -m outputs/TensoIR_Synthetic/hotdog/refgs --eval -w --lambda_mask_entropy 0.05

CUDA_VISIBLE_DEVICES=6 python train.py -s data/TensoIR_Synthetic/hotdog --eval -m outputs/TensoIR_Synthetic/hotdog/irgs  --iterations 20000 --start_checkpoint_refgs outputs/TensoIR_Synthetic/hotdog/refgs/chkpnt50000.pth --envmap_resolution 128 --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 --diffuse_sample_num 256 --envmap_cubemap_lr 0.01 --lambda_light_smooth 0.0005 --init_roughness_value 0.6 --lambda_light 0.1 --light_t_min 0.05 --train_ray

CUDA_VISIBLE_DEVICES=6 python render.py -m outputs/TensoIR_Synthetic/hotdog/irgs --eval --diffuse_sample_num 512 --no_save --no_lpips
CUDA_VISIBLE_DEVICES=6 python compute_albedo_scale_tensoir.py -m outputs/TensoIR_Synthetic/hotdog/irgs
CUDA_VISIBLE_DEVICES=6 python eval_material_tensoir.py -m outputs/TensoIR_Synthetic/hotdog/irgs --no_save --no_lpips --albedo_rescale 2
CUDA_VISIBLE_DEVICES=6 python eval_relighting_tensoir.py -m outputs/TensoIR_Synthetic/hotdog/irgs --diffuse_sample_num 512 --light_sample_num 256 --albedo_rescale 2 -e light   

# lego
CUDA_VISIBLE_DEVICES=7 python train_refgaussian.py -s data/TensoIR_Synthetic/lego -m outputs/TensoIR_Synthetic/lego/refgs --eval -w --lambda_mask_entropy 0.05

CUDA_VISIBLE_DEVICES=7 python train.py -s data/TensoIR_Synthetic/lego --eval -m outputs/TensoIR_Synthetic/lego/irgs  --iterations 20000 --start_checkpoint_refgs outputs/TensoIR_Synthetic/lego/refgs/chkpnt50000.pth --envmap_resolution 128 --lambda_base_color_smooth 2 --lambda_roughness_smooth 0.1 --diffuse_sample_num 256 --envmap_cubemap_lr 0.01 --lambda_light_smooth 0.05 --init_roughness_value 0.8 --lambda_light 0.5 --train_ray

CUDA_VISIBLE_DEVICES=7 python render.py -m outputs/TensoIR_Synthetic/lego/irgs --eval --diffuse_sample_num 512 --no_save --no_lpips
CUDA_VISIBLE_DEVICES=7 python compute_albedo_scale_tensoir.py -m outputs/TensoIR_Synthetic/lego/irgs
CUDA_VISIBLE_DEVICES=7 python eval_material_tensoir.py -m outputs/TensoIR_Synthetic/lego/irgs --no_save --no_lpips --albedo_rescale 2
CUDA_VISIBLE_DEVICES=7 python eval_relighting_tensoir.py -m outputs/TensoIR_Synthetic/lego/irgs --diffuse_sample_num 512 --light_sample_num 256 --albedo_rescale 2 -e light   
