# field_transport_lab

描述实验室与现网终端特征分布，保存基准任务成绩和目标人群快照。

当前公开命令是 `inspect`（核对数据与字段分布）和 `transport fit` / `transport apply`
（从来源样本到目标人群的可迁移完成率估计）。输入和输出使用 UTF-8 JSON。
示例数据均为本地虚构记录，不连接终端、供应方或个人账户。

## 安装与运行

使用 Python 3.11.2，依赖精确版本在 `requirements.txt`。依赖可预先装好；运行和测试不执行下载。

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall -q field_transport_lab
python3 -m unittest discover -s tests -v
python3 -m field_transport_lab inspect --input fixtures/transport.json --output /tmp/transport-profile.json

# 拟合迁移产物；默认同时写出 /tmp/transport-fit.json.artifact.json
python3 -m field_transport_lab.transport fit \
  --input fixtures/transport.json --output /tmp/transport-fit.json

# 换目标区域时只应用产物，不读取新区域结局、不重新拟合
python3 -m field_transport_lab.transport apply \
  --input fixtures/next_region.json --output /tmp/transport-next.json \
  --artifact /tmp/transport-fit.json.artifact.json
```

`fit` 支持 `--input`、`--output`、`--artifact`、`--min-source-cell`、`--trim-cap`、`--z`；
`apply` 支持 `--input`、`--output`、`--artifact`。结果固定包含 `target_coverage`、
`estimates`、`balance`、`sensitivity`、`validation`。`apply` 不读取任何目标结局，
沿用产物中的支持规则与来源格率。

正常运行会写出完整结果文件，标准输出给出保存位置。格式冲突、数据边界错误或缺失关键字段会以非零状态退出；不要将失败输出当作空业务结果。复用临时目录前应清理同名输出与数据库，以免混淆不同运行。

## 数据与目录

`field_transport_lab/population.py` 提供 `inspect` 观测计算；`field_transport_lab/support.py`
提供支持判定、测量含义核对与格子聚合；`field_transport_lab/transport.py` 提供
`fit`/`apply`；`field_transport_lab/io.py` 负责严格 JSON 读取和原子输出；
`fixtures/` 是可重放的演练记录；`tests/` 验证领域边界和输入输出行为；
`docs/transport.md` 记录迁移口径与假设边界。

source 包含 completed，target 只含决策前协变量与 population_weight。validation 的
selected_probability 可能为空，表示复测选择概率未知。measurement_versions 用
field、version、meaning 标识字段测量含义，不能只比较 JSON 类型。

演练规模可在普通 CPU 上完成，运行说明不要求外部服务或真实设备。输出中的未定义值使用 JSON null，不用零替代。涉及版本的输入应保留原文件，方便复查当时的事实。
