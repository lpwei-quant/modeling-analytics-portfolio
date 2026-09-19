# 数据表与字段说明

所有利润均为 **2024—2030 年合计名义人民币元**。`mu` 为亩，`jin` 为斤。情景利润属于 **Simulated**，方案和规范化基础属于 **Derived**。物理土地面积为 1213 亩，两方案各年的多季累计种植面积合计为 9338 亩次，不要混用。

| 文件 | 行数 / 主键 | 内容 |
|---|---|---|
| `q2_final_plan.csv` | 694；`plot,crop_id,year,season` | Q2 最终方案的正面积单元；`area_mu` 为种植面积，未列单元按 0 处理 |
| `q3_final_plan.csv` | 679；同上 | Q3 最终方案的正面积单元；原数值字符串保留，没有四舍五入 |
| `q3_common_paths_summary.csv` | 4；`environment,surplus_rule,plan` | 原 Q3 环境下两个方案 × 两种超产规则的冻结汇总 |
| `q3_common_paths_profits.csv` | 3000；`scenario`，连续 0—2999 | 半价超产规则下两方案在相同路径的利润 |
| `q3_sensitivity_summary.csv` | 36；`dependence_strength,elasticity_scale,surplus_rule,plan` | 原样保存的冻结敏感性汇总 |
| `input_basis.json` | 1 个对象，`schema_version=1` | 54 个地块、41 个作物、125 条完整化经济参数、87 条历史种植记录、47 个作物季次需求代理 |
| `frozen_reference.json` | 1 个对象 | 历史评价种子、3000 路径设计、情景数组指纹与方案结构差异 |

## 图表常用字段

- `q3_common_paths_profits.csv`：`q2_frozen_profit_yuan`、`q3_final_profit_yuan` 必须按相同 `scenario` 配对；其差是同路径 Q2−Q3 利润，不是不同环境或不同种子之差。
- `mean_profit_yuan`：所有路径利润平均值；`standard_deviation_yuan` 为样本标准差，`ddof=1`。
- `cvar_90_yuan`：下尾 10% 平均利润（等权 3000 路径即最差 300 条），高者更好。
- `quantile_05_yuan`、`quantile_10_yuan`、`quantile_20_yuan`：利润分位数，不是置信区间。
- `surplus_rule`：`half_price_surplus` 是超产按 50% 价格出售；`waste_all_surplus` 是超产零收入。后者是固定方案的压力测试，未重新优化。
- `dependence_strength`：`weak/baseline/strong`；`elasticity_scale`：`zero/baseline/strong`；`elasticity_multiplier` 对应 `0/1/1.5`。各组含两个规则与两个方案。
- `mean_cost_yuan`、`mean_production_jin`、`mean_normal_sales_jin`、`mean_surplus_jin`：按路径先核算后求平均。
- 汇总中的 `probability_*` 是有限模拟样本频率；`planted_area_mu` 是累计亩次；`active_capacity_mu` 与 `idle_area_mu` 也是累计口径。接近零的闲置残差来自浮点误差。

## 基础 JSON

`plots`、`crops`、`parameters`、`plantings_2023` 使用冻结 `q1_data.py` 数据类字段。价格区间端点、亩产与成本保留单位后缀；参数记录带 `source_excel_row` 和 `provenance`。智慧大棚第一季经济参数按附件注释沿用普通大棚第一季，属于有来源的推导。

`expected_sales_proxy` 的键为 `crop_id,season`，`quantity_jin` 来自 2023 理论生产量。数值是派生计算，把它作为需求是建模假设。`source_hashes` 仅记录两个附件哈希，不含用户本地路径。

公开 CSV 由冻结表筛选/透视而来，保留数值字符串；转化规则、原表 SHA-256 和公开表 SHA-256 均在项目 [provenance.json](../provenance.json)。两份方案各省略了原 7434 行中的零面积行，独立可行性函数按缺失为零处理。
