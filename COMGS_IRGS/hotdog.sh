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

CUDA_VISIBLE_DEVICES=2 python render_sop.py \
  -s /mnt/store/fd/project/StaticReconstruction/dataset/TensoIR_Synthetic/hotdog \
  -m /mnt/store/fd/project/StaticReconstruction/VirtualRelight/COMGS_IRGS/outputs/TensoIR_Synthetic/hotdog/irgs_sop_new_init_v5 \
  --skip_train