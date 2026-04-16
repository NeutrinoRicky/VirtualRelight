--envmap_representation latlong版本的trace:
CUDA_VISIBLE_DEVICES=2 python train.py -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/hotdog --eval -m ./outputs/TensoIR_Synthetic/hotdog/irgs_octa_env --iterations 20000 --start_checkpoint_refgs /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/hotdog/refgs/chkpnt50000.pth --envmap_resolution 128 --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 --diffuse_sample_num 256 --envmap_cubemap_lr 0.01 --lambda_light_smooth 0.0005 --init_roughness_value 0.6 --lambda_light 0.1 --light_t_min 0.05 --train_ray --init_metallic_value 0.1 --envmap_representation octahedral 

CUDA_VISIBLE_DEVICES=0 python render.py -m ./outputs/TensoIR_Synthetic/hotdog/irgs_octa_env --eval --diffuse_sample_num 256 --skip_train
 --first_k 10

CUDA_VISIBLE_DEVICES=3 python train.py -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego --eval -m ./outputs/TensoIR_Synthetic/lego/irgs_octa_env  -w --iterations 20000 --start_checkpoint_refgs /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/lego/refgs/chkpnt50000.pth --envmap_resolution 128 --lambda_base_color_smooth 2 --lambda_roughness_smooth 0.1 --diffuse_sample_num 256 --envmap_cubemap_lr 0.01 --lambda_light_smooth 0.05 --init_roughness_value 0.8 --lambda_light 0.5 --train_ray


CUDA_VISIBLE_DEVICES=3 python render.py -m ./outputs/TensoIR_Synthetic/lego/irgs_octa_env --eval --diffuse_sample_num 512 --skip_train

CUDA_VISIBLE_DEVICES=3 python render.py -m ./outputs/TensoIR_Synthetic/armadillo/irgs_latlong_env_new_para --eval --diffuse_sample_num 256 --skip_train

CUDA_VISIBLE_DEVICES=3 python render.py -m ./outputs/TensoIR_Synthetic/lego/irgs_octa_env --eval --diffuse_sample_num 512 --skip_train --first_k 10

### SOP
#### 初始化 
CUDA_VISIBLE_DEVICES=1 python SOP/phase1_initializer_point_cloud.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/hotdog \
  -m ./outputs/TensoIR_Synthetic/hotdog/irgs_sop \
  --checkpoint /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/hotdog/refgs/chkpnt50000.pth \
  --offset_scale 0.01 \
  --consistency_min_views 5 \
  --probe_min_distance_factor 0.005 \
  --normal_orientation_mode view_vote

灰色 surface_fused_clean
橙色 probe_surface_samples
红色 probe_offset_points
蓝色 probe_normals_lineset
灰色 surface_fused_clean.ply
这是 fused 出来的表面点云，作为几何参考。

橙色 probe_surface_samples.ply
这是被选出来作为 SOP anchor 的表面点，应该贴在灰色表面上。

红色 probe_offset_points.ply
这是实际 SOP probe 点，应该在橙色点的 normal 外侧一点点。

蓝色 probe_normals_lineset.ply
这是从红色 probe 沿 normal 方向画出来的小线段，用来看 normal 是否朝外。

只看 灰+橙：确认 probe surface sample 是否贴表面
只看 橙+红：确认 offset 是否合理
只看 红+蓝：确认 SOP normal 是否朝外
四个一起看：找穿模、跨面、薄结构泄露

CUDA_VISIBLE_DEVICES=1 python SOP/init_sop_query_textures.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/hotdog \
  -m ./outputs/TensoIR_Synthetic/hotdog/irgs_sop \
  --checkpoint /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/hotdog/refgs/chkpnt50000.pth \
  --probe_file ./outputs/TensoIR_Synthetic/hotdog/irgs_sop/SOP_phase1/probe_init_data.npz \
  --output_dir ./outputs/TensoIR_Synthetic/hotdog/irgs_sop/SOP_query_init

#### lego
CUDA_VISIBLE_DEVICES=1 python train_stage2_sop.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./outputs/TensoIR_Synthetic/lego/irgs_sop_new_para_with_new_loss_v5 \
  --start_checkpoint_refgs /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/lego/refgs/chkpnt50000.pth \
  --sop_init ./outputs/TensoIR_Synthetic/lego/irgs_sop/SOP_query_init/sop_query_init.pt \
  --iterations 2000 \
  --cuda_mem_debug_iters 0 \
  --diffuse_sample_num 256 \
  --visualize_every 100 \
  --lambda_lam 0 \
  --lambda_base_color_smooth 2 \
  --lambda_roughness_smooth 0.1 \
  --lambda_light_smooth 0.05 \
  --init_roughness_value 0.8 \
  --lambda_light 0.5 \
  --init_metallic_value 0.1 \
  --envmap_cubemap_lr 0.01 \
  --lambda_sops 1 \
  --black_background

CUDA_VISIBLE_DEVICES=0 python render_sop.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m /mnt/store/fd/project/StaticReconstruction/VirtualRelight/COMGS_IRGS/outputs/TensoIR_Synthetic/lego/irgs_sop_new_para_with_new_loss_v5 \
  --skip_train

#### armadillo
irgs_sop_new_para_with_new_loss_v3: 真正的 Hammersley sampling with random rotation
irgs_sop_new_para_with_new_loss_v4: 加入真正的loss_sops
CUDA_VISIBLE_DEVICES=2 python train_stage2_sop.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/armadillo \
  -m ./outputs/TensoIR_Synthetic/armadillo/irgs_sop_new_para_with_new_loss_v5 \
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
  --envmap_cubemap_lr 0.01 \
  --lambda_sops 1 \
  --black_background

CUDA_VISIBLE_DEVICES=2 python render_sop.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/armadillo \
  -m /mnt/store/fd/project/StaticReconstruction/VirtualRelight/COMGS_IRGS/outputs/TensoIR_Synthetic/armadillo/irgs_sop_new_para_with_new_loss_v5 \
  --skip_train

#### ficus
CUDA_VISIBLE_DEVICES=1 python train_stage2_sop.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/ficus \
  -m ./outputs/TensoIR_Synthetic/ficus/irgs_sop_new_para_with_new_loss_v4 \
  --start_checkpoint_refgs /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/ficus/refgs/chkpnt50000.pth \
  --sop_init ./outputs/TensoIR_Synthetic/ficus/irgs_sop/SOP_query_init/sop_query_init.pt \
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
  --envmap_cubemap_lr 0.01 \
  --lambda_sops 1 \
  --black_background

CUDA_VISIBLE_DEVICES=0 python render_sop.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/ficus \
  -m /mnt/store/fd/project/StaticReconstruction/VirtualRelight/COMGS_IRGS/outputs/TensoIR_Synthetic/ficus/irgs_sop_new_para_with_new_loss_v4 \
  --skip_train

#### hotdog
irgs_sop_new_init_v1: 修改为使用0.015偏移作为初始化
irgs_sop_new_init_v2: 修改为light_t_min偏移0.05
irgs_sop_new_init_v3: 推翻前面两个,重新选取sop,设定最小距离
irgs_sop_new_init_v4: v3的最小距离很烂,不是对probe自己做的,现在v4才是正确版(但愿)
irgs_sop_new_init_v5: 排查了探针的摆放,一部分探针位置不对,位于盘子底层,重新修改好了

CUDA_VISIBLE_DEVICES=2 python train_stage2_sop.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/hotdog \
  -m ./outputs/TensoIR_Synthetic/hotdog/irgs_sop_new_init_v5 \
  --start_checkpoint_refgs /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/hotdog/refgs/chkpnt50000.pth \
  --sop_init ./outputs/TensoIR_Synthetic/hotdog/irgs_sop/SOP_query_init/sop_query_init.pt \
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
  --envmap_cubemap_lr 0.01 \
  --lambda_sops 1 \
  --black_background

CUDA_VISIBLE_DEVICES=0 python render_sop.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/hotdog \
  -m /mnt/store/fd/project/StaticReconstruction/VirtualRelight/COMGS_IRGS/outputs/TensoIR_Synthetic/hotdog/irgs_sop_new_init_v5 \
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

COMGS SOP v1
armadillo 37.75 0.982 0.037
ficus 32.36 0.985 0.017
hotdog 34.57 0.972 0.042
lego 33.10 0.957 0.047
average 34.45 0.974 0.036

COMGS SOP v4
armadillo PSNR 39.195120067596434 SSIM 0.97838208258152 LPIPS 0.044929707236588
hotdog PSNR 34.02367551803589 SSIM 0.9686657294631005 LPIPS 0.059061929415911436
ficus PSNR 36.04720733642578 SSIM 0.9832488492131233 LPIPS 0.021972426502034067
lego PSNR 32.090149822235105 SSIM 0.9450870189070701 LPIPS 0.055171186979860065


这个版本的roughness还是不太对 难道说Hammersley采样对于Roughness的优化有这么大的影响?
CUDA_VISIBLE_DEVICES=2 python train.py -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/armadillo --eval -m ./outputs/TensoIR_Synthetic/armadillo/irgs_latlong_env_new_para --iterations 20000 --start_checkpoint_refgs /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/armadillo/refgs/chkpnt50000.pth --envmap_resolution 128 --lambda_base_color_smooth 2 --lambda_roughness_smooth 2 --diffuse_sample_num 256 --envmap_cubemap_lr 0.01 --lambda_light_smooth 0.0005 --init_roughness_value 0.6 --lambda_light 0.1 --train_ray --init_metallic_value 0.1 --envmap_representation latlong
