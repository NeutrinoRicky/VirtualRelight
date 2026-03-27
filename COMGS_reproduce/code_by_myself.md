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

SOP 初始化
CUDA_VISIBLE_DEVICES=1 python SOP/phase1_initializer.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/BlendMVS/bull \
  -m ./output/BlendMVS/bull \
  --checkpoint ./output/BlendMVS/bull/chkpnt_best.pth \
  --object_filter_mode weight_and_mask \
  --skip_mesh_export \
  --mask_erosion_radius 8 \
  --fusion_voxel_factor 0.00025 \

CUDA_VISIBLE_DEVICES=1 python SOP/debug_single_view_pointcloud.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/BlendMVS/bull \
  -m ./output/BlendMVS/bull \
  --checkpoint ./output/BlendMVS/bull/chkpnt_best.pth \
  --frame_name 00000001 \
  --mask_erosion_radius 4