# 从官方输入运行完整计算

该目录的 `model/`、`configs/`、`run.py`、`verify.py`、`export.py` 保留最终 v17 支撑包的计算逻辑。`import_inputs.py` 是作品集新增的数据导入入口。原程序与公开副本的对应哈希见上一级 `provenance.json`。

## 环境与数据

使用 Python 3.12，在作品集根目录安装 `requirements-lock.txt`。四份官方输入与五份空结果模板不随仓库重新分发。

1. 从[2026 官方题目发布页](https://www.mcm.edu.cn/html_cn/node/27b6e148f8113f09b0269f64a02629fb.html)获取[官方题包](https://www.mcm.edu.cn/upload_cn/CUMCM2026Problems.zip)，解压 C 题附件。
2. 在本目录运行 `python import_inputs.py "解压后的目录"`。导入器对照 `INPUTS.json` 的 SHA-256；任一文件不匹配就停止，不自动接受修订版数据。
3. 输入复制到被 Git 忽略的 `problem/` 和 `templates/`；不要把它们加入提交。

```shell
python run.py --list
python run.py --case q1
python verify.py --scope small
python run.py --case all --workers 4
python verify.py
python export.py
```

`all` 会重新计算 14 种 Q1 情形、共同一月预热、34 条 334 日轨迹及参数扫描的独立预热。耗时取决于机器，`--workers 1` 可降低内存使用。Q4-3 使用同一输出目录的 Q3 选型记录，首次完整使用运行 `all`。

已有输出只在输入与源码哈希一致时复用。独立重跑可用 `--out output_new`，并向核验和导表程序传递相同输出路径。`--stop` 只用于调试，未完成全年轨迹不能导出为正式结果。

## 本次作品集验证的边界

根目录的验证说明与 `../verification/` 报告本次实际执行范围。历史全量运行证据不能冒充本次全量重跑。快速入口 `python ../verify_representative.py` 直接用公开派生表重新求解 Q1，并核对物理账本和年度汇总。

原程序结果单位为区间 kWh、内部库存 kWh、元。Q3 与 Q4-3 总现金等于原计划费、调整费、紧急购电费之和，计划表和调整表的合计不能再次相加。年度比较是已说明量测条件下的回顾性策略比较，不是前瞻选型保证。
