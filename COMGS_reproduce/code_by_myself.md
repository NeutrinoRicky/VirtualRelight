conda activate surfel_splatting
cd /mnt/store/fd/project/StaticReconstruction/VirtualRelight/COMGS_reproduce

CUDA_VISIBLE_DEVICES=0 python train_stage1_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TNT/Ignatius \
  -m ./output/tnt/Ignatius \
  --lambda_d2n 0.05 \
  --lambda_mask 0.05 \
  --save_gbuffers_every 1000

CUDA_VISIBLE_DEVICES=0 python train_stage1_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/BlendMVS/bull \
  -m ./output/BlendMVS/bull \
  --lambda_d2n 0.05 \
  --lambda_mask 0.05 \
  --save_gbuffers_every 1000 \
  --use_mask_loss

CUDA_VISIBLE_DEVICES=0 python render_stage1_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/BlendMVS/bull \
  -m ./output/BlendMVS/bull \
  --iteration 30000

CUDA_VISIBLE_DEVICES=1 python train_stage2_trace_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/BlendMVS/bull \
  -m ./output/BlendMVS/bull_stage2_trace_irgs \
  --stage1_ckpt ./output/BlendMVS/bull/chkpnt_best.pth \
  --use_mask_loss \
  --lambda_lam 0.001 \
  --lambda_d2n 0.05 \
  --lambda_mask 0.05 \
  --num_shading_samples 128 \
  --save_debug_every 500 \
  --iterations 5000 \
  <!-- --max_trace_points  8192 -->

CUDA_VISIBLE_DEVICES=1 python render_stage2_trace_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/BlendMVS/bull \
  -m ./output/BlendMVS/bull_stage2_trace_irgs \
  --checkpoint ./output/BlendMVS/bull_stage2_trace/object_step2_trace.ckpt \
  --export_mask_mode render 

### SOP 初始化一阶段: SOP初始化在表面(还是从mesh出发了，直接从点云出发有点困难)
CUDA_VISIBLE_DEVICES=1 python SOP/phase1_initializer.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/BlendMVS/bull \
  -m ./output/BlendMVS/bull \
  --checkpoint ./output/BlendMVS/bull/chkpnt_best.pth \
  --object_filter_mode weight_and_mask \
  --skip_mesh_export \
  --mask_erosion_radius 8 \
  --fusion_voxel_factor 0.00025 \

CUDA_VISIBLE_DEVICES=1 python SOP/phase1_initializer.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/BlendMVS/bull \
  -m ./output/BlendMVS/bull \
  --checkpoint ./output/BlendMVS/bull/chkpnt_best.pth \
  --probe_source mesh_largest \
  --target_num_probes 5000 \
  --mesh_surface_sample_count 25000

CUDA_VISIBLE_DEVICES=1 python SOP/debug_single_view_pointcloud.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/BlendMVS/bull \
  -m ./output/BlendMVS/bull \
  --checkpoint ./output/BlendMVS/bull/chkpnt_best.pth \
  --frame_name 00000001 \
  --mask_erosion_radius 4

### SOP 初始化二阶段: 将SOP用光追初始化完成
CUDA_VISIBLE_DEVICES=0 python SOP/init_sop_query_textures.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/BlendMVS/bull \
  -m ./output/BlendMVS/bull \
  --checkpoint ./output/BlendMVS/bull/chkpnt_best.pth \
  --probe_file ./output/BlendMVS/bull/SOP_phase1/probe_offset_points.ply \
  --trace_backend auto \
  --tex_h 16 \
  --tex_w 16、

### SOP 初始化三阶段: 正式训练
CUDA_VISIBLE_DEVICES=0 python train_stage2_sop_decomposition.py \
  -m ./output/BlendMVS/bull \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/BlendMVS/bull \
  --stage1_ckpt ./output/BlendMVS/bull/chkpnt_best.pth \
  --sop_init ./output/BlendMVS/bull/SOP_query_init/sop_query_init.pt

### SOP 初始化三阶段SOP渲染
CUDA_VISIBLE_DEVICES=0 python render_stage2_sop_comgs.py \
  -m ./output/BlendMVS/bull \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/BlendMVS/bull \
  --checkpoint ./output/BlendMVS/bull/object_step2_sop.ckpt \
  --split train \
  --profile_efficiency \
  --profile_efficiency_per_view

  CUDA_VISIBLE_DEVICES=0 python render_stage2_sop_importance_sample_comgs.py \
  -m ./output/BlendMVS/bull \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/BlendMVS/bull \
  --checkpoint ./output/BlendMVS/bull/object_step2_sop.ckpt \
  --split train \
  --profile_efficiency \
  --profile_efficiency_per_view

render_stage2_sop_comgs.py 的实现方式是(效率不加)：

render_multitarget(...) 渲染出 G-buffer
recover_shading_points(...) 恢复表面点
sample_hemisphere_hammersley(...) 生成采样方向
envmap(lightdirs) 得到 direct light
query_sops_directional(...) 从 SOP 查询 indirect light 和 occlusion
integrate_incident_radiance(...) 做 BRDF 积分得到最终 PBR 图