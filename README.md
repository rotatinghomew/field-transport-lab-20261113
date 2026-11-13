# field_transport_lab

描述实验室与现网终端特征分布，保存基准任务成绩和目标人群快照。

当前公开命令是 `inspect`，用于核对数据和查看已记录的观测，不对观测结果作超出数据支持的推断。输入和输出使用 UTF-8 JSON。示例数据均为本地虚构记录，不连接终端、供应方或个人账户。

## 安装与运行

使用 Python 3.11.2，依赖精确版本在 `requirements.txt`。依赖可预先装好；运行和测试不执行下载。

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall -q field_transport_lab
python3 -m unittest discover -s tests -v
python3 -m field_transport_lab inspect --input fixtures/transport.json --output /tmp/transport-profile.json
```

正常运行会写出完整结果文件，标准输出给出保存位置。格式冲突、数据边界错误或缺失关键字段会以非零状态退出；不要将失败输出当作空业务结果。复用临时目录前应清理同名输出与数据库，以免混淆不同运行。

## 数据与目录

`field_transport_lab/population.py` 提供当前观测计算；`field_transport_lab/io.py` 负责严格 JSON 读取和原子输出；`fixtures/` 是可重放的演练记录；`tests/` 验证领域边界和输入输出行为。

source 包含 completed，target 只含决策前协变量与 population_weight。validation 的 selected_probability 可能为空，表示复测选择概率未知。measurement_versions 用 field、version、meaning 标识字段测量含义，不能只比较 JSON 类型。

演练规模可在普通 CPU 上完成，运行说明不要求外部服务或真实设备。输出中的未定义值使用 JSON null，不用零替代。涉及版本的输入应保留原文件，方便复查当时的事实。
