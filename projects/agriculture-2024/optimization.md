# 完整历史计算入口与运行条件

本页保留研究时的实际程序，帮助深入查看模型和求解过程；本次作品集没有重新运行全部优化。已经实际完成的固定方案复算请用[快速复现入口](reproducibility.md)。

`src/` 和 `scripts/` 保留原论文支持包的 18 个 Python 文件，并补入原研究的 `create_q2_freeze_manifest_v1.py` 与 `run_q2_deadline_closure.py`。所有原程序逐字节保留，统一登记在 [provenance.json](provenance.json)；新增打包入口不修改优化目标、参数或历史检查。`historical-inputs/` 的 Q1 决策登记与销量代理表是历史输入，其中的阶段状态按当时语境理解，不是本次新增审批。

## 从官方附件准备隔离工作区

从[官方来源](data-source.md)获取两份附件，安装仓库根的锁定依赖。从仓库根运行：

```shell
python projects/agriculture-2024/scripts/prepare_optimization.py --attachment-1 "附件1.xlsx的本地路径" --attachment-2 "附件2.xlsx的本地路径"
```

脚本先检查两个输入 SHA-256，再建立被 Git 忽略的 `.reproduction/full-model/`，复制必需源码、真实历史输入和冻结方案，并实际调用原读取器验证 54 个地块、41 种作物。已有工作区会拒绝覆盖。输入不会写入原研究目录，也不会进入公开提交。

在该工作区中可以运行原核心入口：

```shell
python scripts/reproduce_core.py --paths 3000
python scripts/reproduce_core.py --paths 3000 --solve-q1
```

第一条重新评价 Q2 独立环境和 Q3 共同环境下的冻结方案；第二条同时求解两种超产规则的 Q1 MILP。求解有原程序规定的时限和 gap 条件，不保证在任意机器上都达到成功状态。原 `run_q1.py` 另包含 Excel 输出，需要自行从官方题包把 `result1_1.xlsx`、`result1_2.xlsx` 放到工作区 `outputs/`。

## Q2 / Q3 历史优化路线

| 入口 | 实际作用与先决条件 |
|---|---|
| `scripts/run_q2.py` | 原 Q2 候选求解及评价入口；要求完整 Q1 冻结清单和其列出的历史文件。默认 `all` 保留早期流程，不能把它误当作最终已放弃 θ 前沿的推荐复现命令 |
| `scripts/run_q2_deadline_closure.py` | 最终 Q2 收尾的双上界证书及评价；在已有期望收益候选、收据和 Q1 冻结状态上继续，`--stage certificate` / `evaluation` 明确区分工作 |
| `scripts/run_q3.py` | Q3 候选、证书与共同路径评价；要求 Q2 冻结清单及其列出的原历史产物 |

这些入口保留真实冻结校验，因此**只有官方附件，不能恢复整个历史 Q2/Q3 执行状态**。公开包没有复制全部旧会话、阶段材料和候选搜索产物，也没有伪造或重新签发历史冻结清单。需要继续原历史优化时，应在有合法完整存档的独立副本中恢复它们所要求的 `outputs/` 等文件，并核对已有哈希；不可通过跳过检查把新结果标为旧冻结结果。

原优化模型函数本身也保留在 `q1_model.py`、`q2_model.py`、`q3_model.py` 中，便于理解和另行开展新的实验。新的计算属于新运行，不自动继承历史采纳状态。

## 本次确认

- 补入的原文件与源存档逐字节相同。
- 原入口的导入闭包检查通过；有命令行解析的入口 `--help` 可用，未启动耗时求解。
- 用两份本地官方输入完成隔离工作区准备，原读取器得到 54 个地块和 41 种作物。
- 这些检查只证明文件闭包、参数入口与基础加载，不代表完整优化或历史证书已经重跑。
