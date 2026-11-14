# 记录口径

描述实验室与现网终端特征分布，保存基准任务成绩和目标人群快照。

source 包含 completed，target 只含决策前协变量与 population_weight。validation 的 selected_probability 可能为空，表示复测选择概率未知。measurement_versions 用 field、version、meaning 标识字段测量含义，不能只比较 JSON 类型。

数据字段采用稳定的英文标识符。编号是业务身份，不由输入数组中的位置决定。未到达的结果与已经观察到的否定结果是不同事实。货币、时间和统计单位应按字段约定解释，不在读取时默默变换。

当前命令的输出只解释当前公开计算，不承诺示例覆盖全部业务情形。批量重放应使用新的输出位置，避免人工查看上一次运行留下的文件。

`fixtures/transport.json` 是拟合用的来源/目标/复测记录；`fixtures/next_region.json` 是只含上线前协变量与人口权重的下一区域快照，用来演示 `transport apply`。它保留了来源从未招募的 `entry` 硬件层，演示不会因为换区域就把这类人群算进点估计。迁移口径与假设边界见 `docs/transport.md`。
