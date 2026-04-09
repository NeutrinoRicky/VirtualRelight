auto
按顺序自动尝试 irgs_adapter -> irgs_native -> open3d_mesh，谁先能建起来就用谁，逻辑在 tracing_comgs.py (line 401)。所以它不是一个独立后端，而是“自动回退模式”。想保证实验可复现，最好别用 auto。

irgs
现在其实只是 irgs_adapter 的别名。代码里 requested in ("auto", "irgs", "irgs_adapter") 都走同一分支，见 tracing_comgs.py (line 407)。也就是说当前仓库里，irgs 和 irgs_adapter 没有实际区别。

irgs_adapter
直接用 IRGS2DGaussianTraceAdapter 的原始输出，见 tracing_comgs.py (line 151) 和 irgs_trace_adapter.py (line 19)。
它的关键行为是：

occlusion = alpha
incident_radiance = color
在这个分支里 envmap、secondary_num_samples、randomized_secondary 被直接丢掉了，见 tracing_comgs.py (line 186)
所以这是“直接相信 IRGS tracer 返回的软 alpha 和 color”的模式，遮挡是软的，不是硬命中。

irgs_native
名字有点迷惑。它底层其实也还是先调同一个 IRGS2DGaussianTraceAdapter 拿 hit/material，见 tracing_comgs.py (line 115)。
真正不同的是它不用 adapter 直接返回的 alpha/color 当最终结果，而是走 BaseTraceBackend.trace 的通用逻辑：

occlusion = hit_mask，是 0/1 的硬遮挡
incident_radiance 会基于命中的材质和 envmap 再做一次 shade_secondary_points
所以这个模式下 secondary_num_samples 才真正起作用，行为更接近 “先求交，再根据材质和环境光重算入射辐射”。

你可以把它简单记成：

irgs_adapter / irgs：直接吃 IRGS tracer 的 alpha + color
irgs_native：用 IRGS tracer 只做求交，光照再按 COMGS 这边的逻辑重算
auto：先试前者，失败再退后者，再退 Open3D
再结合你的训练式子看，loss 里最终用的是
incident_radiance = (1 - occlusion) * direct_radiance + trace_outputs["incident_radiance"]，见 losses_comgs_stage2_trace.py (line 189)。
所以 irgs_adapter 和 irgs_native 的差别，本质上就在于这两个量的定义不同。

