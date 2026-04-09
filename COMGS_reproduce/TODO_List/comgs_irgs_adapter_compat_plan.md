# COMGS 适配 IRGS Adapter 语义的兼容式修改方案

本文档给出一版**尽量不改已有代码主体、优先通过新增文件与新增函数实现**的修改方案，用于让：

- `COMGS_reproduce/utils/irgs_trace_adapter.py`
- `COMGS_reproduce/utils/losses_comgs_stage2_trace.py`
- `COMGS_reproduce/render_stage2_trace_comgs.py`

更接近 IRGS 原版 `render_ir -> rendering_equation -> pc.trace(...)` 的语义。

---

## 目标

本方案的核心目标不是“一步到位把 COMGS 彻底改成 IRGS”，而是先建立一个**IRGS-like compatibility path**，用于隔离并验证以下问题：

1. `irgs_adapter` 返回的 `trace_outputs["color"]` 是否能作为 IRGS 语义下的 `local_incident_lights`
2. COMGS 当前结果差，究竟主要来自：
   - 采样策略差异
   - trace color 语义差异
   - metallic-roughness BRDF 解释差异
3. 在**不破坏现有 COMGS stage2 代码**的前提下，新增一条可切换的 IRGS-compatible 路径

---

# 总体原则

## 原则 1：不直接重写旧逻辑
不要直接大改这三个文件中的现有主函数。  
应优先：

- 新建兼容模块
- 新增可选参数
- 新增分支函数
- 在原入口处仅增加少量接线代码

## 原则 2：保留 COMGS 现有路径
保留当前默认行为不变。也就是说：

- 不开启新 flag 时，仍走原来的 COMGS stage2 tracing 逻辑
- 开启兼容 flag 时，才走 IRGS-like 路径

## 原则 3：先对齐 IRGS 的 `base_color + roughness` 语义
第一阶段不要让 `metallic` 参与 trace feature 的语义。  
即：

- trace feature 只传 `base_color/albedo + roughness`
- metallic 暂时固定为 0，或者仅在最终 BRDF 中可选参与
- 先验证 trace + shading 对齐 IRGS 是否能提升 NVS

---

# 需要新增的文件

建议新增以下文件：

1. `COMGS_reproduce/utils/irgs_compat_trace_utils.py`
2. `COMGS_reproduce/utils/irgs_compat_sampling.py`
3. `COMGS_reproduce/utils/irgs_compat_shading.py`

---

# 一、修改 `irgs_trace_adapter.py`

---

## 目标

当前 `IRGS2DGaussianTraceAdapter.trace()` 默认把 feature 设为：

- `albedo`
- `roughness`
- `metallic`

这会把 trace 命中材质语义切到 COMGS 自己的 PBR 体系。  
为了更接近 IRGS，需要让 adapter 支持一种**IRGS-compatible feature layout**：

- 前 3 维：`base_color`
- 第 4 维：`roughness`

并且返回时也能兼容这套 4 维 feature，而不是强依赖第 5 维 metallic。

---

## 不要删除旧逻辑

### 保留现有：
- `IRGS2DGaussianTraceAdapter`
- `IRGSAdapterTraceBackend`
- 现有 `trace()` 默认行为

### 新增：
1. 一个新的 helper，用于构造 IRGS-compatible feature
2. 一个新的 `trace_irgs_compat(...)`
3. 或者在现有 `trace(...)` 中增加 `feature_mode` 参数，但默认仍为旧模式

---

## 具体建议

### 方案 A（推荐）：新增 helper + 新参数

在 `irgs_trace_adapter.py` 中新增：

#### 1. 新增 helper：`_build_trace_features(...)`

新增一个函数，例如：

- `feature_mode="comgs_pbr"`：旧行为
- `feature_mode="irgs_base_rough"`：新行为

语义如下：

### `feature_mode="comgs_pbr"`
返回：
- albedo [3]
- roughness [1]
- metallic [1]

shape = `[N, 5]`

### `feature_mode="irgs_base_rough"`
返回：
- albedo/base_color [3]
- roughness [1]

shape = `[N, 4]`

注意：
- 这里直接复用 `gaussians.get_albedo` 作为 IRGS-like `base_color`
- 不需要新增新的高斯属性名
- 只是语义上把 `albedo` 当作 `base_color`

---

#### 2. 修改 `IRGS2DGaussianTraceAdapter.trace(...)`

新增可选参数：

- `feature_mode: str = "comgs_pbr"`

逻辑改为：

- 如果 `features is None`
  - 调 `_build_trace_features(self.gaussians, feature_mode=feature_mode)`

这样不会影响旧调用。

---

#### 3. 返回值兼容 4 维 / 5 维 feature

当前代码里写死了：

- `albedo = feature[:, :3]`
- `roughness = feature[:, 3:4]`
- `metallic = feature[:, 4:5]`

这在 4 维 IRGS-compatible 模式下会出问题。  
所以需要改成：

- 前 3 维始终解释为 `base_color/albedo`
- 第 4 维解释为 `roughness`
- 若 feature 维数 >= 5，则取 metallic
- 否则 metallic 返回全 0

注意：  
不要改字段名，继续返回：

- `"albedo"`
- `"roughness"`
- `"metallic"`

这样下游兼容性最好。

---

## 建议新增代码结构

建议在本文件新增：

- `_build_trace_features(...)`
- `_split_trace_feature_outputs(...)`

其中：

### `_split_trace_feature_outputs(feature_tensor, device, dtype)`
统一做：
- albedo/base_color 提取
- roughness 提取
- metallic 提取（缺省补 0）

这样可避免重复逻辑。

---

## 建议新增的 backend 包装

为了避免污染现有 `IRGSAdapterTraceBackend`，建议**新增一个 backend 类**：

### `IRGSAdapterCompatTraceBackend(BaseTraceBackend)`

行为与 `IRGSAdapterTraceBackend` 基本一样，但内部调用 adapter 时固定使用：

- `feature_mode="irgs_base_rough"`

这样：
- 原 `irgs_adapter` 不变
- 新增 `irgs_adapter_compat`
- 便于命令行切换与 A/B test

---

## build_trace_backend 接线建议

在 `utils/tracing_comgs.py` 中：

### 不改旧 backend 名称逻辑
保留：
- `irgs_adapter`
- `irgs_native`
- `open3d`

### 新增一个 backend 选项
新增：

- `irgs_adapter_compat`

在构建 backend 时：
- `irgs_adapter` -> 原逻辑
- `irgs_adapter_compat` -> 新 backend 类

---

# 二、修改 `losses_comgs_stage2_trace.py`

---

## 目标

新增一条 **IRGS-compatible stage2 shading path**，使其在训练时可选地：

1. 使用 IRGS-like trace feature 语义
2. 使用 IRGS-like direct + indirect 组合
3. 使用 IRGS-like BRDF（`base_color/pi + fixed-F0 GGX`）
4. 采样策略支持从“仅半球采样”扩展到“半球 + env importance 混合采样”

---

## 不要直接改掉旧函数主体

当前主函数是：

- `compute_stage2_trace_loss(...)`

不要直接把里面原来的 COMGS 路径重写。  
建议改成：

- 保留原函数
- 在函数内部根据新 flag 分流
- 或者把旧逻辑抽成 `_compute_stage2_trace_loss_comgs(...)`
- 新增 `_compute_stage2_trace_loss_irgs_compat(...)`

推荐后者，结构更清晰。

---

## 建议新增文件：`utils/irgs_compat_shading.py`

把 IRGS-compatible 公式单独放到新文件，避免污染旧 loss 文件。

建议新增以下函数：

### 1. `ggx_specular_irgs_compat(...)`
目的：
- 直接复用/改写现有 IRGS 风格 GGX
- Fresnel 固定为 0.04
- 不使用 metallic 去混 F0

### 2. `integrate_incident_radiance_irgs_compat(...)`
输入：
- `base_color`
- `roughness`
- `normals`
- `viewdirs`
- `lightdirs`
- `incident_radiance`
- `incident_weights_or_areas`

输出：
- `diffuse`
- `specular`
- `shaded`

语义严格贴近 IRGS：

- `f_d = base_color / pi`
- `f_s = GGX_specular(..., fresnel=0.04)`
- `transport = incident_radiance * sample_weight * n_dot_l`
- 最终对样本维做平均或求和（与权重定义一致）

### 3. `compose_incident_lights_irgs_compat(...)`
输入：
- `trace_alpha`
- `trace_color`
- `direct_env_radiance`

输出：
- `incident_lights = (1 - alpha) * direct_env_radiance + trace_color`

这个函数存在的意义是显式把 IRGS 的语义写清楚，不要散落在 loss 主体里。

---

## 建议新增文件：`utils/irgs_compat_sampling.py`

建议新增：

### 1. `sample_incident_dirs_diffuse_irgs_compat(...)`
包装现有半球采样逻辑，语义命名成 IRGS 风格。

可直接内部调用现有：
- `sample_hemisphere_hammersley(...)`
或新增 Fibonacci 版本，但第一阶段无需强求。

### 2. `sample_incident_dirs_env_irgs_compat(...)`
新增从 envmap 重要性采样方向的函数。

如果当前 envmap 类已经有：
- `sample_light_directions(...)`
- `light_pdf(...)`

就直接复用。  
如果没有，就在该新文件中加一个 wrapper。

### 3. `sample_incident_dirs_mixture_irgs_compat(...)`
这是重点函数。

输入：
- normals
- envmap
- diffuse_sample_num
- light_sample_num
- randomized

输出建议包含：
- `incident_dirs`
- `incident_pdf`
- `sample_weight`

语义：
- 从半球采样一部分方向
- 从 env importance 采样一部分方向
- 合并为一批方向
- 用 mixture pdf 做权重修正

如果你不想第一版就把 mixture estimator 写得很复杂，也可以先返回：

- `incident_dirs`
- `incident_areas = 1 / pdf`

只要和后续积分接口匹配即可。

---

## 训练损失里新增 IRGS-compatible 分支

在 `losses_comgs_stage2_trace.py` 中：

### 新增配置参数
建议新增以下参数到 `compute_stage2_trace_loss(...)`：

- `shading_mode: str = "comgs_pbr"`
- `trace_feature_mode: str = "comgs_pbr"`
- `use_irgs_mixture_sampling: bool = False`
- `diffuse_sample_num: int = 128`
- `light_sample_num: int = 0`
- `force_metallic_zero_in_irgs_mode: bool = True`

默认都保持旧行为。

---

### 新增分支逻辑

#### 当 `shading_mode == "comgs_pbr"`
保持旧逻辑不变

#### 当 `shading_mode == "irgs_compat"`
走新增分支：

1. 从 rasterized map 中拿：
   - `albedo` 作为 `base_color`
   - `roughness`
2. metallic：
   - 若 `force_metallic_zero_in_irgs_mode=True`，则直接设为 0，但只用于可视化，不参与 IRGS BRDF
3. 采样方向：
   - 若 `use_irgs_mixture_sampling=False`
     - 仅用原半球采样
   - 若为 True
     - 调 `sample_incident_dirs_mixture_irgs_compat(...)`
4. 调 tracer：
   - 使用 `irgs_adapter_compat`
   - 传入 `feature_mode="irgs_base_rough"`
5. 组合 incident lights：
   - `incident = (1-alpha)*direct_env + trace_color`
6. 积分：
   - `integrate_incident_radiance_irgs_compat(...)`
7. 输出：
   - `pbr_render` 虽然字段名可保持不变，但语义上这是 IRGS-like render

---

## 训练日志建议

为了对比方便，建议新增若干统计量：

- `trace_alpha_mean`
- `trace_color_mean`
- `incident_direct_mean`
- `incident_indirect_mean`
- `irgs_compat_diffuse_mean`
- `irgs_compat_specular_mean`

这样便于判断：
- trace color 是否过暗
- alpha 是否异常偏高
- indirect 是否被压制

---

# 三、修改 `render_stage2_trace_comgs.py`

---

## 目标

新增一个**渲染时的 IRGS-compatible 输出路径**，保证训练和渲染使用同一种语义。

---

## 不要改掉原 `render_stage2_trace_view(...)`

建议：

- 保留现有 `render_stage2_trace_view(...)`
- 新增一个：
  - `render_stage2_trace_view_irgs_compat(...)`

然后在外层根据参数切换。

---

## 新增参数

在渲染脚本中增加命令行参数：

- `--shading_mode`
  - choices: `["comgs_pbr", "irgs_compat"]`
- `--trace_feature_mode`
  - choices: `["comgs_pbr", "irgs_base_rough"]`
- `--use_irgs_mixture_sampling`
- `--diffuse_sample_num`
- `--light_sample_num`
- `--force_metallic_zero_in_irgs_mode`

默认：
- `shading_mode="comgs_pbr"`
- `trace_feature_mode="comgs_pbr"`

这样不会影响旧脚本。

---

## 新增函数：`render_stage2_trace_view_irgs_compat(...)`

建议逻辑：

1. 调 `render_multitarget(...)` 取：
   - render
   - albedo
   - roughness
   - metallic
   - normal
   - depth
2. 从深度恢复 points
3. 对 valid points 采样 incident dirs：
   - 若 `use_irgs_mixture_sampling=False`，走原半球采样
   - 否则走 mixture sampling
4. 调 trace backend：
   - 要求 backend 是 `irgs_adapter_compat`
5. 组合：
   - `incident = (1 - alpha) * direct_env + trace_color`
6. 用 IRGS-compatible BRDF 积分：
   - diffuse + specular
7. 输出图：
   - `pbr_render`（字段名保留）
   - `trace_direct`
   - `trace_indirect`
   - `trace_occlusion`
   - `trace_alpha`
   - `trace_color_raw`
   - `irgs_diffuse`
   - `irgs_specular`

注意：
- `trace_color_raw` 很有用，必须单独导出，方便判断 tracer 返回的局部颜色究竟是什么量级与色调

---

## 输出可视化建议

在 `save_stage2_trace_outputs(...)` 中，新增保存：

- `{view_name}_trace_color_raw.png`
- `{view_name}_trace_alpha.png`
- `{view_name}_irgs_diffuse.png`
- `{view_name}_irgs_specular.png`

所有 HDR 项仍建议 tonemap 后保存 PNG。

---

# 四、关于 metallic 的兼容策略

---

## 第一阶段建议：不要让 metallic 进入 trace feature
这是整个方案中最重要的一条工程建议。

### 原因
IRGS 原版的 trace 语义并不是以 metallic-roughness 为中心构造的。  
如果你一开始就把 metallic 也混进 trace feature：

- 你无法判断 trace color 变差是不是 metallic 导致的
- 你无法判断是 tracer 语义偏了，还是 BRDF 解释偏了

### 建议做法
#### 在 `irgs_adapter_compat` 模式下：
- trace feature 只传：
  - base_color
  - roughness

#### metallic 的使用：
- 第一阶段：
  - 训练和渲染都不参与 IRGS-compatible BRDF
  - 或者直接强制为 0
- 第二阶段：
  - 若 IRGS-compatible 路径验证有效，再逐步把 metallic 放回最终 BRDF，而不是放回 trace feature

---

# 五、建议新增的最小命令行实验矩阵

---

## 实验 A：完全保守的 IRGS-compatible 验证
目标：验证 adapter + IRGS-like shading 是否能显著缩小和 IRGS 的差距

推荐设置：

- `trace_backend=irgs_adapter_compat`
- `shading_mode=irgs_compat`
- `trace_feature_mode=irgs_base_rough`
- `use_irgs_mixture_sampling=false`
- `force_metallic_zero_in_irgs_mode=true`

如果这一步已经比当前 COMGS 好很多，说明主要问题在：
- metallic-roughness 解释
- 以及 COMGS 现有 shading path

---

## 实验 B：仅替换采样策略
在实验 A 基础上，开启：

- `use_irgs_mixture_sampling=true`
- `diffuse_sample_num=...`
- `light_sample_num=...`

目标：验证环境光重要性采样是否继续带来改善。

---

## 实验 C：最后再恢复 metallic
在实验 B 基础上：

- IRGS-compatible trace feature 仍保持 `base+rough`
- 只在最终 BRDF 中恢复 metallic-roughness

目标：判断 metallic 到底是不是拖累项。

---

# 六、Codex 实施时的具体要求

请按以下约束修改：

1. **不要删除或重写原有主函数**
2. 所有新增逻辑优先放在新文件中
3. 原文件中只做：
   - import 新文件
   - 新增可选参数
   - 新增少量分支调用
4. 默认参数下，原 COMGS 行为必须完全不变
5. 新增 backend / shading_mode 必须能通过命令行切换
6. 所有新增导出图像命名必须清晰，便于 A/B 对比
7. 所有新函数需要加简洁注释，说明“这是 IRGS-compatible compatibility path，不是替代原 COMGS path”

---

# 七、建议给 Codex 的实现顺序

请严格按以下顺序实现，以便每一步都可单独测试。

## Step 1
新增：
- `utils/irgs_compat_shading.py`
- `utils/irgs_compat_sampling.py`

先不要接主流程。

## Step 2
修改 `irgs_trace_adapter.py`：
- 新增 `feature_mode`
- 新增 `_build_trace_features`
- 新增 `_split_trace_feature_outputs`
- 新增 `IRGSAdapterCompatTraceBackend`

先确保：
- 默认旧逻辑不变
- 新 backend 可构造成功

## Step 3
修改 `utils/tracing_comgs.py`
- 注册 `irgs_adapter_compat`

先做最小 smoke test。

## Step 4
修改 `utils/losses_comgs_stage2_trace.py`
- 把旧逻辑尽量包进 `_compute_stage2_trace_loss_comgs(...)`
- 新增 `_compute_stage2_trace_loss_irgs_compat(...)`
- 在总入口按 `shading_mode` 分发

## Step 5
修改 `render_stage2_trace_comgs.py`
- 新增 `render_stage2_trace_view_irgs_compat(...)`
- 加命令行参数
- 新增可视化导出

## Step 6
做最小验证：
- 默认参数下旧结果不变
- 新参数下能顺利训练 / 渲染一小批图
- 导出的 `trace_color_raw / trace_alpha / irgs_diffuse / irgs_specular` 看起来合理

---

# 八、你需要重点检查的验收点

Codex 改完后，请重点检查：

1. `irgs_adapter_compat` 返回的：
   - `albedo`
   - `roughness`
   - `metallic`
   shape 是否正确
2. 当 `feature_mode="irgs_base_rough"` 时：
   - metallic 是否真的自动补零
3. `trace_color_raw` 是否明显偏暗 / 偏灰 / 偏花
4. `trace_alpha` 是否大面积接近 1
5. 开启 `irgs_compat` 后，NVS 是否比当前 COMGS 默认路径明显改善
6. 开启 mixture sampling 后，是否进一步改善高光与间接光区域

---

# 九、一个非常重要的判断标准

如果你做完本方案后发现：

- `irgs_adapter_compat + irgs_compat shading`
明显优于当前 COMGS 默认路径，

那么几乎就可以确认：

**当前 COMGS NVS 差的主要原因，不是 tracer 本身，而是你现在对 tracer 输出的解释方式（尤其是 metallic-roughness + 仅半球采样这套组合）出了偏差。**

反过来，如果这样改完仍然很差，那么才值得继续深挖：

- trace color 本身的物理语义
- surfel tracer 对 SH color 的返回是否满足你对“局部辐射”的预期
- 几何 / 法线 / 深度恢复是否本身就有系统偏差

---

# 十、交付要求

请 Codex 最终交付：

1. 所有新增/修改文件列表
2. 每个文件的变更摘要
3. 新增的命令行参数说明
4. 一组最小可运行命令：
   - 训练命令
   - 渲染命令
5. 明确说明：
   - 默认行为未变
   - 新逻辑仅在新 flag 启用时生效
