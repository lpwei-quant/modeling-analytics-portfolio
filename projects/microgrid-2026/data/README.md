# 数据来源与字段

原题来源：[2026 年 CUMCM 官方发布页](https://www.mcm.edu.cn/html_cn/node/27b6e148f8113f09b0269f64a02629fb.html)。官方附件为 Observed 输入，原始 XLSX 不在公开仓库中重新分发，获取和哈希检查见[复现说明](../reproduction/README.md)。

这里的表格来自最终 v17 计算和绘图材料，为 **Derived**。它们描述题目数据下的模型输出，不是现实场站收益。复制来源与哈希见[provenance.json](../provenance.json)，汇总提取依据见[source_notes.json](source_notes.json)。

| 文件 | 粒度 | 用途与单位 |
|---|---|---|
| `F04_q1_dispatch.csv` | 144 个十分钟区间 | 原附件功率 kW、区间能量 kWh、价格元/kWh、调度与现金元；含原始行号和时间标签 |
| `q1_reference.json` | 题给代表日 | 冻结费用、无储能参照、库存与能量汇总 |
| `F06_q2_annual.csv` | 策略 × 334 天 | 日计划费、紧急费、总现金元与日末库存 kWh |
| `strategy_summary.csv` | 13 组完整策略 | 同一 334 日范围；计划费、调整费、紧急费、总现金元，首末库存 kWh |
| `F10_q3_policies.csv` | Q3 配置及对照 | 费用万元、末库存 kWh；绘图标签沿用冻结版本 |
| `F12_q4_costs.csv` | Q2→4-2、Q3→4-3 | 均价、协方差、重新决策费用分解元；起始和最终策略末库存 kWh |

现金恒等式是 `plan_cost_yuan + adjustment_cost_yuan + emergency_cost_yuan = total_cash_yuan`。Q2 的调整费为零。具体合同结算定义见原 `model/contracts.py` 与论文，不把同一费用重复累计。

**Assumed / Decision：**单程效率各 0.9、区间时间映射、当期量测近似、动态价格可知性等是明确的题意解释或建模条件。**Simulated：**独立构造小例用于检查数学和实现，不是新增观测。
