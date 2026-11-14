# Fig. 1 总体架构与合成图绘制方案 v2（已按 percnet_v3.yaml + registry.py 逐项核对）

> 替代 v1《框架图绘制方案》。主结果模型 = `percnet_v3_wx`（yaml 默认 + variant 覆盖 {`use_exo_channels:False, weather_context:True`}）。本文件每条画法都有代码/配置出处，改图前先核对本表。

---

## 0. 配置核对结论（决定画什么、不画什么）

| 事实 | 配置出处 | 对绘图的影响 |
|---|---|---|
| ① **EESM 启用**：v3 分支对全部 `percnet_v3*` 硬置 `use_eesm=True`，wx variant 不覆盖 | registry.py:174 | 谱增强**必须画进 Fig.1**，但按"继承组件"细灰框，不进设计徽章；消融表无 EESM 行→正文不得声称其贡献 |
| ② **degree response 关闭**：`degree_response = True AND use_exo` → wx 下为 False | registry.py:169-176 + percnet_v3.py:171 | **不画**温度响应模块（属路径 B），v1 方案结论维持 |
| ③ **上下文只进融合、不碰分位头**：`wx_quantile=False`（v3_wxq 才开） | registry.py:173 | 框架图中 45 维上下文**只有一条出边→融合 MLP**；合成图 b 扇形图的题注**禁止**写"天气条件化的区间" |
| ④ **上下文从原始 x 统计，不经 RevIN**：`_weather_context(x)` 用未归一化列 3–11 | percnet_v3.py:397, 369 | 上下文框的入边直接从"输入块"引出、绕过 RevIN；RevIN 框只覆盖负荷 3 列（气象列直通保持静态尺度） |
| ⑤ 特征路径：`feat = ctx_proj([mean_t(季节专家第一层特征), GRU隐层均值])`（36+12→36）；`e_h` 零初始化（H×36）；分位数中心 = μ（不 detach）；σ 头 bias 初始化 5.0 | percnet_v3.py:413-429 | 图上要画 feat 支路（从季节专家与 GRU 各引一根细线）；"centered on μ"标注在分位数头 |
| ⑥ 周期相位来自日历列 12–16 的 atan2 反解（no_grad） | percnet_v3.py:292-307 | 日历列只连"周期队列寻址"，不连编码器 |
| ⑦ 融合输入维度 = 9（三流×负荷 3 通道）+ 45 = 54 | percnet_v3.py:263-265 | 融合框标 `54→64→32→3, zero-init` |
| ⑧ 训练侧参数（供图注/§5）：Adam lr 8e-4、wd 1e-5、clip 5、50 epochs、验证 MAPE plateau(0.5/2)、batch 64、prnn_dropout 0.3 | registry.py UNIFIED_RECIPE + v3 分支 | 与 v6 正文 §5 数字一致，直接引用 |

---

## 1. Fig. 1 总体架构图（双栏宽 17.5 cm，高 8–9 cm，单一矢量 PDF）

### 构图原则（维持 v1）
"展示路由，不展示网络"。三视觉等级：**细灰框=继承**（RevIN、分解、线性专家、补丁 GRU、EESM、镜像谱）/ **粗蓝框=本文设计**（上下文向量、残差融合、逐域条件分位数头、联合损失）/ **浅块=张量**。设计组件挂 R1/R3/R4 圆徽；R2 由周期/锚的"减-加"闭环箭头表达。

### 泳道①：输入
```
观测历史窗 x ∈ R^{168×16}   [observed only, 无未来行]
 ├─ 列0–2  负荷 E/C/H ──────────────→ 走主链
 ├─ 列3–11 外生 9 列（气象6+DOY-COS+CMBMBBTU+GHG）
 │        ✕ 不进入任何流编码器（"the network is never asked to forecast the weather"）
 └─ 列12–15 日历 sin/cos ──→ 仅连周期队列寻址（细虚线）
```

### 泳道②：先验剥离（上排）+ 上下文（下排）
- 负荷路径：`RevIN(loads-only)` → 历史侧 `−cycle(daily24×3+weekly168×3)` → `−last-value anchor(detach)` → 残差 r。
- **EESM（依事实①补画，灰框）**：挂在 r 进入三流之前：`rFFT → 镜像谱掩码(85×3) → irFFT 增强`；输出侧对应 `− 复数频率线性(过散分量)`。注脚小字 "FESTRNN-style, inherited"。
- 上下文框（粗蓝，徽章 R1）：`trailing-24h × 9 列 × {mean,min,max,last,last−mean} = 45 标量`。**入边从泳道①输入块直接引出（事实④），出边恰好一条→泳道④融合（事实③）**。框下钉数字：`L=72h: +0.4 MAE · weather-only: +4.3 MAE（§6.2/6.3）`。

### 泳道③：三流编码器（灰框，并行三条水平带）
| 流 | 标注 |
|---|---|
| 季节线性专家 | `168→36→H, LeakyReLU, channel-independent` |
| 趋势线性专家 | 同构 |
| 补丁 GRU | `seg24, d12, dropout 0.3；一-shot 解码；pos⊕chan emb` |
- 三条带右端汇聚 `stack = s + t + loc`（加号节点，标注 "epoch-0 forecast"）。
- 链的准确表述写进图例：`component-wise experts, parallel forward; "chain" = exclusive supervision (boosting-style, stop-gradient)`——与正文术语一致，避免读成串行拓扑。

### 泳道④：融合与概率头（两粗蓝框）
- **融合（R3）**：`μ = stack[:, :3] ⊕ Fusion([s,t,loc,c])`，Fusion 标 `54→64→32→3, zero-init`；stack 主干粗实线直穿，MLP 侧路挂在旁边。侧注 `−residual form: +16.0 MAE`。
- **分位数头（R4）**：`feat(36) ⊕ e_h(H×36, zero-init)` → `Linear 36→27`（9 分位×3 载体）+ `σ 头 softplus`；`q_k(h) = μ + W_k φ(feat+e_h)` 直接标注在框内。feat 支路按事实⑤画两根细入边（季节专家特征均值、GRU 隐层均值→48→36）。扇形示意小图标**随 h 增宽**；对照虚影"单尺度塌缩（PICP 0.48→0.33, DeepARQ）"。
- 头下损失徽章：`MAPE + pinball(9) + NLL + CVaR₀.₉ + mono + dec(×10)`；注 `换纯 MAPE: +5.8 MAE, 方差×2`。

### 泳道⑤：输出与决策接口
按代码顺序三还原：`+cycle +anchor`（与泳道②成对的回加虚线，R2 闭环）→ `RevIN⁻¹(loads)` → 输出 `μ ∈ R^{H×3}, Q ∈ R^{H×3×9}`，旁挂恒等式框 `CMB = 0.003412·E + 0.012·C + 1.0·H`（零参数）。
末端报童小图标（<10%）：`Q(τ*) → 备用承诺 → −14.3%…−70.0%（§6.5）`。

### 底部总结栏
`26,890 params · 0.259 MMac/24h窗（44% FESTRNN）· observed-only · PICP₉₀ 0.899–0.907 by construction`
（可选分模块参数量注一行：先验 <600 · 专家 ~13.9k · 融合 ~5.7k · 头+嵌入 ~2.0k · EESM ~2.5k · GRU ~1.6k；数字画前用代码实测校准）

---

## 2. 合成图（拟合曲线+扇形+热图）——新布局规范

**布局**：a 上半部（2 行×3 列）、b 左下角、c 右下角。全图矢量、最小字号 ≥6.5 pt。**全部数据从 v6 口径重导，禁止沿用旧图数值**（旧图基于作废的 v3 全通道实验，且图例拼写为 PECR-Net）。

### 子图 a（上半，2 行×3 列）
- 列 = Electric (kW) / Cooling (CHWTON) / Heating (MBtu/h)；行 = 24 h / 96 h。
- 每格曲线集合与 **Table 1 的 8 模型**对齐：Observed（棕实线）、PERC-Net (joint) 蓝实线、PERC-Net (point) 蓝虚线、FESTRNN、DLinear、iTransformer、Crossformer、segRNN、Mamba、FiLM（线型自定但须统一用到全图）。旧图 9 条线里没有 segRNN/含旧变体的，须重导。
- x = Time (h) 0–96（或窗长），灰竖点线标日分界；y 只最左列；**图例统一 "PERC-Net"**，单行置于 a 区底部。

### 子图 b（左下）
- Cooling 96 步分位数扇形：Observed 黑实线、Median (τ=0.5) 红线、9 条 τ=0.05–0.95 蓝色系曲线 + 右色标。
- **题注只允许写 horizon-conditioned**（事实③：wx_quantile=False，区间不随天气条件化）；示例窗口取 PERC-Net official seed 2024 / h96 测试段，标注"illustrative segment"。
- 宽约左下 60%，高约 4 cm。

### 子图 c（右下）
- 校准偏差热图 12×9：行 = 3 载体×{24,48,72,96}，列 = 9 分位（刻度仅 0.05/0.50/0.95）；格值 = coverage − nominal，蓝(−0.1)–白–红(+0.1)，格内两位小数 + 底部横色标。
- 数据必须用 v6 `T4_prob` 口径重新计算逐分位偏差（旧值 −0.14/+0.08 作废）。
- 尺寸约 4×4.5 cm，与 b 底边对齐。

### 合成图整幅题注草案（英）
> Fig. X. (a) Fitted 24 h (top) and 96 h (bottom) forecasts per carrier on representative test segments; curve set matches Table 1. (b) 96-step quantile fan for cooling (seed 2024, illustrative): intervals widen with horizon by construction. (c) Per-carrier, per-horizon calibration deviation (coverage − nominal) across nine quantiles; aggregated PICP₉₀ = 0.899–0.907.

---

## 3. 其余图与正文联动（简单）

| 图 | 数据源 | 一条关键约束 |
|---|---|---|
| Fig.2 数据特征（可沿用内容、矢量重导） | 原始记录 | 相关系数用已核验值：WBT→E/C/H = 0.81/0.84/−0.76；CMB 恒等式残差 ≤0.08 |
| Fig.3 三路径对比 | `T2_paths` | B 行标注"含谱增强/温度响应的全通道路径"（那才是 642.4 的配置），C 行即 Fig.1 |
| Fig.4 覆盖率–预测域 raw vs CQR | `T4_prob` | 三模型实/虚线 + 0.90 参考带；数字以图为准与 Table 4 逐格一致 |
| Fig.5 决策价值 | `T5_decision` | log-x cu/co，4 时域曲线 ± 误差带（std ≤2.5 pp） |
| 前沿散点（独立 Fig.6 或并入 Fig.3） | `T1_point`×`T7_budget` | h24 MAE vs MMac，Pareto 前沿高亮 PERC-Net/FESTRNN；**图号悬空问题在 v6 [VERIFY] 中一并解决** |

---

## 4. 画前自检清单

- [ ] 外生 9 列无任何箭头进入三流编码器；上下文出边恰好 1 条、指向融合
- [ ] 上下文入边绕过 RevIN（从原始 x）；RevIN 框只盖负荷 3 列
- [ ] EESM 画出且为灰框继承级；degree response 与 12 通道部件 0 出现
- [ ] 分位头入边 = feat ⊕ e_h（无天气支路）；图例中 "chain" 按 exclusive-supervision 措辞
- [ ] 周期/锚"历史减、输出加"成对；恒等式框系数与 §3.2 一致
- [ ] 合成图曲线集合与 Table 1 对齐；图例 PERC-Net；扇形题注无"天气条件化"
- [ ] 全部数字与 `T2/T4/T5` 表逐格一致（621.1 消融口径与 621.6 头条口径不混用）
- [ ] 每张图单一矢量 PDF，Figure_1…独立命名；单栏 ≥1063 px / 全幅 ≥2244 px 等效
