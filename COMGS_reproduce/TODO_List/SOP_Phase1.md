Phase 1：只做 SOP 几何位置初始化，不训练

这一步的目标非常纯粹：

得到一批位置合理、不会 light leak、查询稳定的 SOP 中心点和法线。

TODO 1.1：整理可复用输入

你需要从你现有 stage1 版管线里拿到：

多视角相机参数
当前 object 的 geometry buffers
depth
normal
mask / weight

这一步不需要 SOP 纹理，只要几何信息。

TODO 1.2：从多视角深度恢复表面点云

按论文做法，先对所有 viewpoint 渲染 geometry buffers，再做 multi-view depth fusion，得到稠密表面点云 + 法线。论文这里明确写了这条流程。

你实现上可以先不追求完全复刻他们的 fusion 细节，第一版可以这样做：

对每个训练视角：
从 unbiased depth 反投影到世界坐标
用 normal buffer 一起带出来
只保留 mask 内、alpha/weight 足够大的点
合并所有视角点
用体素下采样 / 半径去重做 clean

也就是说，第一版可以先用“工程替代版的 multi-view fusion”。

TODO 1.3：做 uniform subsample

论文这里是 FPS。

所以你要：

输入：稠密表面点云 P 和法线 N
输出：
K 个 probe anchor
第一版建议：
K = 5000（和论文一致）
如果物体太小，先从 1000 / 2000 开始调试
TODO 1.4：沿法线偏移 probe 位置

论文说得很明确：
采样到的表面点要沿法线轻微偏移，避免 probe 和高斯/表面混在一起导致 light leaking。offset 经验值是物体尺寸的 1%。

所以这里做：

probe_pos = surface_pos + offset * normal
offset = 0.01 * object_extent

这里的 object_extent 建议先定义成：

bbox diagonal，或者
max side length

但整个项目里一定要固定一种定义，不然 offset 会乱。

TODO 1.5：做 probe 初始化可视化

这一步非常重要，先别急着训练。

你至少要做 4 个 debug 可视化：

object 高斯 + probe 点
probe 法线方向箭头
probe 到最近表面的距离分布
probe 是否落在表面内部 / 背面 / 漂太远
TODO 1.6：做“位置质量检查”

给每个 probe 统计：

到最近表面点距离
到最近高斯中心距离
沿法线方向是否朝外
是否有明显 probe 扎进物体内部

这一步的目标不是论文指标，而是排除最常见问题：

probe 太贴表面 → numerical/self-occlusion 问题
probe 太远 → 插值失真
probe 穿进物体内部 → 光照/遮挡全错