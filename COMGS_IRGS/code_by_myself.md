CUDA_VISIBLE_DEVICES=3 python train.py -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/ficus --eval -m ./outputs/TensoIR_Synthetic/ficus/irgs_octa_env  -w --iterations 20000 --start_checkpoint_refgs /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/ficus/refgs/chkpnt50000.pth --envmap_resolution 128 --lambda_base_color_smooth 2 --lambda_roughness_smooth 0.1 --diffuse_sample_num 256 --envmap_cubemap_lr 0.01 --lambda_light_smooth 0.005 --init_roughness_value 0.6 --lambda_light 0.1 --train_ray

CUDA_VISIBLE_DEVICES=3 python train.py -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego --eval -m ./outputs/TensoIR_Synthetic/lego/irgs_octa_env  -w --iterations 20000 --start_checkpoint_refgs /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/lego/refgs/chkpnt50000.pth --envmap_resolution 128 --lambda_base_color_smooth 2 --lambda_roughness_smooth 0.1 --diffuse_sample_num 256 --envmap_cubemap_lr 0.01 --lambda_light_smooth 0.05 --init_roughness_value 0.8 --lambda_light 0.5 --train_ray


CUDA_VISIBLE_DEVICES=3 python render.py -m ./outputs/TensoIR_Synthetic/lego/irgs_octa_env --eval --diffuse_sample_num 512 --skip_train
CUDA_VISIBLE_DEVICES=3 python render.py -m ./outputs/TensoIR_Synthetic/armadillo/irgs_octa_env --eval --diffuse_sample_num 512 --skip_train

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

CUDA_VISIBLE_DEVICES=2 python train_stage2_sop.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/hotdog \
  -m ./outputs/TensoIR_Synthetic/hotdog/irgs_sop \
  --start_checkpoint_refgs /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/hotdog/refgs/chkpnt50000.pth \
  --sop_init ./outputs/TensoIR_Synthetic/hotdog/irgs_sop/SOP_query_init/sop_query_init.pt \
  --iterations 2000 \
  --cuda_mem_debug_iters 0 \
  --diffuse_sample_num 128 \
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

COMGS Trace
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

IRGS
armadillo 39.86 0.977 0.047
ficus 35.74 0.981 0.025
hotdog 35.18 0.966 0.052
lego 31.04 0.930 0.074
average 35.45 0.964 0.050
