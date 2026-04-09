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

CUDA_VISIBLE_DEVICES=1 python train_stage1_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./output/TensorIR/lego/stage1 \
  --lambda_d2n 0.05 \
  --lambda_mask 0.05 \
  --save_gbuffers_every 1000 \
  --use_mask_loss \
  --eval

  1阶段渲染:
  CUDA_VISIBLE_DEVICES=0 python render_stage1_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./output/TensorIR/lego/stage1 \
  --iteration -1 \
  --eval

  CUDA_VISIBLE_DEVICES=0 python render.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./output/TensorIR/lego/stage1 \
  --iteration -1 \
  --eval \
  --skip_mesh

  CUDA_VISIBLE_DEVICES=0 python metrics.py \
  -m /mnt/store/fd/project/StaticReconstruction/VirtualRelight/COMGS_reproduce/output/TensorIR/lego/stage1

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

CUDA_VISIBLE_DEVICES=0 python train_stage2_trace_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./output/TensorIR/lego \
  --stage1_ckpt ./output/TensorIR/lego/chkpnt_best.pth \
  --use_mask_loss \
  --lambda_lam 0.001 \
  --lambda_d2n 0.05 \
  --lambda_mask 0.05 \
  --num_shading_samples 128 \
  --save_debug_every 500 \
  --iterations 5000 \
  <!-- --max_trace_points  8192 -->

从 IRGS 的 refgs checkpoint 兼容初始化 Stage2 Trace:
CUDA_VISIBLE_DEVICES=0 python train_stage2_trace_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./output/TensorIR/lego_from_refgs \
  --stage1_ckpt /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/lego/refgs/chkpnt50000.pth \
  --stage1_ckpt_format irgs_refgs \
  --use_mask_loss \
  --lambda_lam 0.001 \
  --lambda_d2n 0.05 \
  --lambda_mask 0.05 \
  --num_shading_samples 128 \
  --save_debug_every 500 \
  --iterations 5000

CUDA_VISIBLE_DEVICES=1 python render_stage2_trace_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./output/TensorIR/lego \
  --checkpoint ./output/TensorIR/lego/object_step2_trace.ckpt \
  --split test \
  --eval \
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

CUDA_VISIBLE_DEVICES=0 python SOP/phase1_initializer.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./output/TensorIR/lego/stage2/trace \
  --checkpoint ./output/TensorIR/lego/chkpnt_best.pth \
  --object_filter_mode weight_and_mask \
  --mask_erosion_radius 8 \
  --fusion_voxel_factor 0.00025 \
  --probe_source mesh_largest \
  --target_num_probes 5000 \
  --mesh_surface_sample_count 25000

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
  --tex_w 16

CUDA_VISIBLE_DEVICES=0 python SOP/init_sop_query_textures.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./output/TensorIR/lego \
  --checkpoint ./output/TensorIR/lego/chkpnt_best.pth \
  --probe_file ./output/TensorIR/lego/SOP_phase1/probe_offset_points.ply \
  --trace_backend auto \
  --tex_h 16 \
  --tex_w 16

### SOP 初始化三阶段: 正式训练
CUDA_VISIBLE_DEVICES=0 python train_stage2_sop_decomposition.py \
  -m ./output/BlendMVS/bull \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/BlendMVS/bull \
  --stage1_ckpt ./output/BlendMVS/bull/chkpnt_best.pth \
  --sop_init ./output/BlendMVS/bull/SOP_query_init/sop_query_init.pt

CUDA_VISIBLE_DEVICES=0 python train_stage2_sop_decomposition.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./output/TensorIR/lego \
  --stage1_ckpt ./output/TensorIR/lego/chkpnt_best.pth \
  --sop_init ./output/TensorIR/lego/SOP_query_init/sop_query_init.pt

### SOP 初始化三阶段SOP渲染
CUDA_VISIBLE_DEVICES=0 python render_stage2_sop_comgs.py \
  -m ./output/BlendMVS/bull \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/BlendMVS/bull \
  --checkpoint ./output/BlendMVS/bull/object_step2_sop.ckpt \
  --split train \
  --profile_efficiency \
  --profile_efficiency_per_view

CUDA_VISIBLE_DEVICES=3 python render_stage2_sop_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./output/TensorIR/lego \
  --checkpoint ./output/TensorIR/lego/object_step2_sop.ckpt \
  --split test \
  --eval \
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

加上mask之后渲染：
CUDA_VISIBLE_DEVICES=0 python render_stage2_sop_importance_sample_object_comgs.py \
  -m ./output/BlendMVS/bull \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/BlendMVS/bull \
  --checkpoint ./output/BlendMVS/bull/object_step2_sop.ckpt \
  --split test \
  --profile_efficiency \
  --profile_efficiency_per_view

CUDA_VISIBLE_DEVICES=0 python render_stage2_sop_importance_sample_object_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/tensorir/lego \
  -m ./output/TensorIR/lego \
  --checkpoint ./output/TensorIR/lego/object_step2_sop.ckpt \
  --split test \
  --eval
  --profile_efficiency \
  --profile_efficiency_per_view

  CUDA_VISIBLE_DEVICES=1 python metrics.py -m /mnt/store/fd/project/StaticReconstruction/VirtualRelight/COMGS_reproduce/output/TensorIR/lego/stage2_trace_render
  CUDA_VISIBLE_DEVICES=0 python metrics.py -m /mnt/store/fd/project/StaticReconstruction/VirtualRelight/COMGS_reproduce/output/TensorIR/lego/stage2_sop_render


### Hotdog
CUDA_VISIBLE_DEVICES=1 python train_stage2_trace_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/hotdog \
  -m ./output/TensorIR/hotdog/stage2/trace \
  --stage1_ckpt /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/hotdog/refgs/chkpnt50000.pth \
  --stage1_ckpt_format irgs_refgs \
  --trace_backend irgs_adapter \
  --freeze_geometry \
  --no_freeze_color \
  --num_shading_samples 256 \
  --trace_num_rays 262144 \
  --trace_bias 0.05 \
  --envmap_lr 0.01 \
  --envmap_height 128 \
  --envmap_width 128 \
  --envmap_init_value 1.5 \
  --lambda_mask 0.01 \
  --lambda_base_color_smooth 2.0 \
  --lambda_roughness_smooth 0.1 \
  --lambda_normal_smooth 0.01 \
  --lambda_light 0.5 \
  --lambda_light_smooth 0.05 \
  --save_debug_every 500 \
  --iterations 5000

CUDA_VISIBLE_DEVICES=1 python render_stage2_trace_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/hotdog \
  -m ./output/TensorIR/hotdog/stage2/trace \
  --checkpoint ./output/TensorIR/hotdog/stage2/trace/object_step2_trace.ckpt \
  --split test \
  --eval \
  --export_mask_mode render 

CUDA_VISIBLE_DEVICES=1 python metrics.py -m /mnt/store/fd/project/StaticReconstruction/VirtualRelight/COMGS_reproduce/output/TensorIR/hotdog/stage2/trace/stage2_trace_render --render_suffix _rgb_raw.png

### Lego: 第二阶段光追，尝试采用RefGS的初始化几何结果
输出为./output/TensorIR/stage2/lego_from_refgs
环境光照可能有问题，要重新train一下
TODO：注意 采样从原来的 Hammersley 改成了 IRGS 的 Fibonacci hemisphere sampling，需要修改回来 metallic的初始化也有问题，需要修改
d2n 改成 IRGS 的 rend_normal 对 surf_normal，并且从 iteration > 1000 才启用，losses_comgs_stage2_trace.py (line 253) 和 train_stage2_trace_comgs.py (line 539)。
light_smooth 改成对当前视角的 env_only 图做 TV，而不是直接对 env 参数图做 TV，losses_comgs_stage2_trace.py (line 74) 和 losses_comgs_stage2_trace.py (line 292)。
render_multitarget 的 normal map 也补成了 rend_normal / alpha 后再 normalize，和 IRGS normal_map 对齐，gaussian_renderer/init.py#L245 (line 245)。
默认 lambda_lam 现在是 0.0，不再额外带一个 IRGS 没有的材质先验，train_stage2_trace_comgs.py (line 537)。


CUDA_VISIBLE_DEVICES=1 python train_stage2_trace_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./output/TensorIR/lego/stage2/trace \
  --stage1_ckpt /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/lego/refgs/chkpnt50000.pth \
  --stage1_ckpt_format irgs_refgs \
  --trace_backend irgs_adapter \
  --freeze_geometry \
  --no_freeze_color \
  --num_shading_samples 256 \
  --trace_num_rays 262144 \
  --trace_bias 0.05 \
  --envmap_lr 0.01 \
  --envmap_height 128 \
  --envmap_width 128 \
  --envmap_init_value 1.5 \
  --lambda_mask 0.01 \
  --lambda_base_color_smooth 2.0 \
  --lambda_roughness_smooth 0.1 \
  --lambda_normal_smooth 0.01 \
  --lambda_light 0.5 \
  --lambda_light_smooth 0.05 \
  --save_debug_every 500 \
  --iterations 5000

CUDA_VISIBLE_DEVICES=3 python train_stage2_trace_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./output/TensorIR/lego/stage2/trace_v2 \
  --stage1_ckpt /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/lego/refgs/chkpnt50000.pth \
  --trace_backend irgs_adapter_compat \
  --shading_mode irgs_compat \
  --trace_feature_mode irgs_base_rough \
  --diffuse_sample_num 128 \
  --iterations 5000 \
  --use_irgs_mixture_sampling \
  --light_sample_num 256 \
  --trace_rebuild_every 1 \



CUDA_VISIBLE_DEVICES=1 python render_stage2_trace_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./output/TensorIR/lego/stage2/trace_v2 \
  --checkpoint ./output/TensorIR/lego/stage2/trace_v2/object_step2_trace.ckpt \
  --split test \
  --eval \
  --export_mask_mode render \

CUDA_VISIBLE_DEVICES=1 python metrics.py -m /mnt/store/fd/project/StaticReconstruction/VirtualRelight/COMGS_reproduce/output/TensorIR/lego/stage2/trace_v2/stage2_trace_render --render_suffix _rgb_raw.png --max_frames 10
SOP 训练 训练效率是上来了,但albedo还是过高,可能环境光的学习率得改
先初始化SOP，再初始化探针材质
CUDA_VISIBLE_DEVICES=1 python SOP/phase1_initializer.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./output/TensorIR/lego/stage2/sop \
  --checkpoint /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/lego/refgs/chkpnt50000.pth \
  --checkpoint_format irgs_refgs \
  --output_dir ./output/TensorIR/lego/stage2/sop \
  --object_filter_mode weight_and_mask \
  --mask_erosion_radius 8 \
  --fusion_voxel_factor 0.00025 \
  --probe_source mesh_largest \
  --target_num_probes 5000 \
  --mesh_surface_sample_count 25000

CUDA_VISIBLE_DEVICES=1 python SOP/init_sop_query_textures.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./output/TensorIR/lego/stage2/sop \
  --checkpoint /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/lego/refgs/chkpnt50000.pth \
  --checkpoint_format irgs_refgs \
  --probe_file ./output/TensorIR/lego/stage2/sop/probe_offset_points.ply \
  --output_dir ./output/TensorIR/lego/stage2/sop \
  --trace_backend auto \
  --tex_h 16 \
  --tex_w 16

CUDA_VISIBLE_DEVICES=2 python train_stage2_sop_decomposition.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./output/TensorIR/lego/stage2/sop \
  --stage1_ckpt /mnt/store/fd/project/StaticReconstruction/IRGS/outputs/TensoIR_Synthetic/lego/refgs/chkpnt50000.pth \
  --stage1_ckpt_format irgs_refgs \
  --sop_init ./output/TensorIR/lego/stage2/sop/sop_query_init.pt \
  --iterations 10000 \
  --trace_backend irgs_adapter \
  --freeze_geometry \
  --no_freeze_color \
  --sop_query_topk 5

SOP render：G-buffer时间压缩, SOP knn缓存
CUDA_VISIBLE_DEVICES=1 python render_stage2_sop_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./output/TensorIR/lego/stage/sop \
  --checkpoint ./output/TensorIR/lego/stage2/sop/object_step2_sop.ckpt \
  --split test \
  --eval \
  --profile_efficiency \
  --profile_efficiency_per_view

SOP 重要性采样
256条光线 top 4的SOP
CUDA_VISIBLE_DEVICES=3 python render_stage2_sop_importance_sample_object_comgs.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/lego \
  -m ./output/TensorIR/lego \
  --checkpoint ./output/TensorIR/lego/object_step2_sop.ckpt \
  --split test \
  --eval \
  --profile_efficiency \
  --profile_efficiency_per_view \
  --disable_sample_jitter

CUDA_VISIBLE_DEVICES=1 python metrics.py -m /mnt/store/fd/project/StaticReconstruction/VirtualRelight/COMGS_reproduce/output/TensorIR/lego/stage2/sop/stage2_sop_render

git push github master