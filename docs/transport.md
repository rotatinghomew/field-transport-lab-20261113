# 迁移分析口径

`transport fit` 把实验室（来源）的完成率迁移到现网快照（目标）的上线前人群。
它回答的是“来源样本能代表目标人群的哪一部分、那部分成绩是多少”，不是重新训练一个
完成率预测器。

## 估计对象与边界

- 可支持部分：目标格子同时满足（1）来源同一协变量组合有足够样本，
  （2）该协变量组合在两边都可交换。只有这部分进入 `primary_transport`、
  `trimmed_transport` 和 `outcome_model_scenario` 的点估计。
- 全人群：只报告 `whole_population_bounds`，其下界为
  `supported_share × 支持区域估计`，上界再加 `(1 - supported_share)`。
  工具**拒绝**给出全人群点估计（`whole_population_point_estimate: 拒绝`），
  因为来源从未覆盖的硬件层（示例中的 `entry`）没有任何结局数据。

## 来源招募限制

`recruitment.excluded_tiers` 是来源招募时明确排除的硬件层。它比数据本身更早、
更硬：即使目标里出现了该层，也不能因为样本量或模型预测就把它算进迁移。
`required_network` 同理，用于记录招募只覆盖某种网络状态的情形。

## 测量版本

`measurement_versions` 的每条记录是 `field + version + meaning`。
相同 JSON 编码**不等于**相同测量含义。示例中两侧 `network` 都写 `normal`/`weak`，
但一侧是“接入信号等级”，另一侧是“十秒内请求丢包等级”。因此：

- 版本含义不一致的字段不进入迁移协变量（`excluded_columns`），
  其边际只列在 `balance.marginals` 中，并标记 `used_for_transport: false`。
- 同名编码造成的偏差会作为一个**错误假设**情景
  （`sensitivity.scenarios[naive_meaning_mismatch]`）报告，只用于估量偏差，
  不作为结论，也不进入全人群边界。
- 行级 `measurement_version` 缺失或没有对应含义记录时，该字段判为不可迁移。

## 权重的含义

在支持区域内，单格权重 = 目标格人口份额 / 来源格样本份额。
`sensitivity` 对权重比设上限（默认 `--trim-cap 2`），并重算估计，用来显示
少数高权重格子对结论的影响。`effective_sample_size` 按权重集中度计算；
示例中 180 行来源支持样本的有效样本约 115，说明 `mid|booking`
（权重比 2.5）对结果影响较大。

## 能检查的与只能做敏感性分析的

可检查（用允许的信息即可核对）：

- 支持判定：来源是否覆盖目标格子、来源格样本量是否达到 `--min-source-cell`；
- 测量含义：同名版本的含义是否一致；
- 边际平衡：目标份额、来源份额与权重比；
- 结果模型：只用来源数据拟合，在 holdout 上独立核对。

只能做敏感性分析（数据里没有可证伪的信息）：

- 未覆盖硬件层（`entry`）的完成率——只能给边界；
- 测量含义不同的字段——不能交换，只能演示偏差；
- 现场复测的选择机制——`selected_probability` 为空的复测只能做情景。

## 现场复测的用途

`validation` 只用 `split == "holdout"` 的行检验，`development` 行不参与拟合、
也不进入检验。两类误差分开报告：

- `transport_error`：holdout 观测率与来源同格支持率之差（迁移误差）；
- `selection_bias`：在同一格子内用 `selected_probability` 校正前后的差，
  以及选择概率已知/未知两组的完成率差（选择偏差）。

二者都不把复测当随机真值：复测是按业务挑选的，且 stderr 不覆盖这种挑选机制。

## 产物与换区域

`fit` 把来源预处理（支持规则、来源格率与样本量、测量版本、指纹）保存为
`<output>.artifact.json`（也可用 `--artifact` 指定）。`apply`：

- 只读取产物与**新快照的上线前协变量与人口权重**；
- 不读取任何目标结局，也不重新拟合；
- 用产物里的支持规则（招募排除、未出现水平、样本量门槛）重新判定支持，
  所以新区域里来源曾覆盖、但拟合目标里没出现的格子仍会得到支持；
- 新快照的测量版本与含义必须与拟合时一致，否则以非零状态退出；
- 报告新区域支持格份额相对拟合时的漂移（`balance.share_drift`）。

这样换目标区域时不会“偷偷”用新区域结局重拟合。
