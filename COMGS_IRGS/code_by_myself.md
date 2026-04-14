--envmap_representation latlong版本的trace:
CUDA_VISIBLE_DEVICES=2 python train.py -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/armadillo --eval -m ./outputs/TensoIR_Synthetic/armadillo/irgs_octa_env_new_para --iterations 20000 --start_checkpoint_refgs /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/armadillo/refgs/chkpnt50000.pth --envmap_resolution 128 --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 --diffuse_sample_num 256 --envmap_cubemap_lr 0.01 --lambda_light_smooth 0.0005 --init_roughness_value 0.6 --lambda_light 0.1 --train_ray --init_metallic_value 0.1 --envmap_representation latlong 

CUDA_VISIBLE_DEVICES=3 python render.py -m ./outputs/TensoIR_Synthetic/armadillo/irgs_octa_env_new_para --eval --diffuse_sample_num 512 --skip_train
 --first_k 10

CUDA_VISIBLE_DEVICES=3 python train.py -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego --eval -m ./outputs/TensoIR_Synthetic/lego/irgs_octa_env  -w --iterations 20000 --start_checkpoint_refgs /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/lego/refgs/chkpnt50000.pth --envmap_resolution 128 --lambda_base_color_smooth 2 --lambda_roughness_smooth 0.1 --diffuse_sample_num 256 --envmap_cubemap_lr 0.01 --lambda_light_smooth 0.05 --init_roughness_value 0.8 --lambda_light 0.5 --train_ray


CUDA_VISIBLE_DEVICES=3 python render.py -m ./outputs/TensoIR_Synthetic/lego/irgs_octa_env --eval --diffuse_sample_num 512 --skip_train

CUDA_VISIBLE_DEVICES=3 python render.py -m ./outputs/TensoIR_Synthetic/armadillo/irgs_latlong_env_new_para --eval --diffuse_sample_num 256 --skip_train

CUDA_VISIBLE_DEVICES=3 python render.py -m ./outputs/TensoIR_Synthetic/lego/irgs_octa_env --eval --diffuse_sample_num 512 --skip_train --first_k 10

### SOP
CUDA_VISIBLE_DEVICES=3 python SOP/phase1_initializer_point_cloud.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/ficus \
  -m ./outputs/TensoIR_Synthetic/ficus/irgs_sop \
  --checkpoint /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/ficus/refgs/chkpnt50000.pth

CUDA_VISIBLE_DEVICES=3 python SOP/init_sop_query_textures.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/ficus \
  -m ./outputs/TensoIR_Synthetic/ficus/irgs_sop \
  --checkpoint /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/ficus/refgs/chkpnt50000.pth \
  --probe_file ./outputs/TensoIR_Synthetic/ficus/irgs_sop/SOP_phase1/probe_init_data.npz \
  --output_dir ./outputs/TensoIR_Synthetic/ficus/irgs_sop/SOP_query_init

#### armadillo
CUDA_VISIBLE_DEVICES=1 python train_stage2_sop.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/armadillo \
  -m ./outputs/TensoIR_Synthetic/armadillo/irgs_sop_new_para_with_new_loss_v2 \
  --start_checkpoint_refgs /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/armadillo/refgs/chkpnt50000.pth \
  --sop_init ./outputs/TensoIR_Synthetic/armadillo/irgs_sop/SOP_query_init/sop_query_init.pt \
  --iterations 2000 \
  --cuda_mem_debug_iters 0 \
  --diffuse_sample_num 256 \
  --visualize_every 100 \
  --lambda_lam 0 \
  --lambda_base_color_smooth 2 \
  --lambda_roughness_smooth 2 \
  --lambda_light_smooth 0.0005 \
  --init_roughness_value 0.6 \
  --lambda_light 0.1 \
  --init_metallic_value 0.1 \
  --envmap_cubemap_lr 0.01

CUDA_VISIBLE_DEVICES=3 python render_sop.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/armadillo \
  -m /mnt/store/fd/project/StaticReconstruction/VirtualRelight/COMGS_IRGS/outputs/TensoIR_Synthetic/armadillo/irgs_sop_new_para_with_new_loss_v2 \
  --skip_train

#### ficus
CUDA_VISIBLE_DEVICES=3 python train_stage2_sop.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/ficus \
  -m ./outputs/TensoIR_Synthetic/ficus/irgs_sop \
  --start_checkpoint_refgs /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/ficus/refgs/chkpnt50000.pth \
  --sop_init ./outputs/TensoIR_Synthetic/ficus/irgs_sop/SOP_query_init/sop_query_init.pt \
  --iterations 2000 \
  --cuda_mem_debug_iters 0 \
  --diffuse_sample_num 128 \
  --visualize_every 100 \
  --lambda_lam 0

CUDA_VISIBLE_DEVICES=3 python render_sop.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/ficus \
  -m /mnt/store/fd/project/StaticReconstruction/VirtualRelight/COMGS_IRGS/outputs/TensoIR_Synthetic/ficus/irgs_sop \
  --skip_train

#### hotdog
CUDA_VISIBLE_DEVICES=2 python train_stage2_sop.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/hotdog \
  -m ./outputs/TensoIR_Synthetic/hotdog/irgs_sop \
  --start_checkpoint_refgs /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/hotdog/refgs/chkpnt50000.pth \
  --sop_init ./outputs/TensoIR_Synthetic/hotdog/irgs_sop/SOP_query_init/sop_query_init.pt \
  --iterations 2000 \
  --cuda_mem_debug_iters 0 \
  --diffuse_sample_num 256 \
  --visualize_every 100 \
  --lambda_lam 0

CUDA_VISIBLE_DEVICES=2 python render_sop.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/hotdog \
  -m /mnt/store/fd/project/StaticReconstruction/VirtualRelight/COMGS_IRGS/outputs/TensoIR_Synthetic/hotdog/irgs_sop \
  --skip_train

gt pbr_render albedo roughness
metallic weight depth normal
depth_normal query_selection query_occlusion query_direct
query_indirect pbr_diffuse pbr_specular error map

IRGS
armadillo 39.86 0.977 0.047
ficus 35.74 0.981 0.025
hotdog 35.18 0.966 0.052
lego 31.04 0.930 0.074
average 35.45 0.964 0.050

only metallic + long env
armadillo 39.8229275894165 0.9774968528747559 0.04441660460084677
armadillo_new_para 39.63
ficus 35.55645191192627 0.9806574696302414 0.02472425905056298
hotdog 35.83521836280823 0.9697868725657464 0.046996580604463815
lego 32.38461189270019 0.9477025073766708 0.05591930773109197

metallic + long env + Hammersley
armadillo 38.005099353790285

metallic + Hammersley + octa
armadillo 39.816475582122806 0.9779640939831734 0.04444860877469182


COMGS Trace(看来不是环境光照的问题, 是metallic的问题啊)
armadillo 38.05 0.981 0.040
ficus 
hotdog 
lego 32.38 0.948 0.056
average

COMGS SOP
armadillo 37.75 0.982 0.037
ficus 32.36 0.985 0.017
hotdog 34.57 0.972 0.042
lego 33.10 0.957 0.047
average 34.45 0.974 0.036


这个版本的roughness还是不太对 难道说Hammersley采样对于Roughness的优化有这么大的影响?
CUDA_VISIBLE_DEVICES=2 python train.py -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/armadillo --eval -m ./outputs/TensoIR_Synthetic/armadillo/irgs_latlong_env_new_para --iterations 20000 --start_checkpoint_refgs /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/armadillo/refgs/chkpnt50000.pth --envmap_resolution 128 --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 --diffuse_sample_num 256 --envmap_cubemap_lr 0.01 --lambda_light_smooth 0.0005 --init_roughness_value 0.6 --lambda_light 0.1 --train_ray --init_metallic_value 0.1 --envmap_representation latlong
