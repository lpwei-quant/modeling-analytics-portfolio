# 固定方案复算

## 环境与命令

项目专用 [requirements.txt](requirements.txt) 记录本次成功运行的 NumPy、Pandas、SciPy 和 OpenPyXL 版本。Python 使用 3.12.14。可使用仓库根目录已建立的虚拟环境；若单独使用本项目，在隔离环境内执行：

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

安装前先激活新环境；Windows 为 `.venv\Scripts\Activate.ps1`，macOS/Linux 为 `source .venv/bin/activate`。所有命令以上述环境中的 `python` 执行。脚本以自身位置确定项目根，既可以从项目目录运行，也可以从仓库根运行 `python projects/agriculture-2024/scripts/reproduce.py`。

默认报告在 `.reproduction/verification.json`，不会改写冻结输入。`--output` 可指定报告位置。本次归档命令为：

```bash
python scripts/reproduce.py --output evidence/verification.json
```

## 实际验证范围

1. 比对公开冻结文件 SHA-256，检查基础规模和本项目内的导入位置。
2. 读取固定 Q2/Q3 种植计划，对容量、适种性、水浇地制度、2023 初始轮作、相邻物理机会轮作和三年豆类累计面积规则逐项验证。
3. 按 Q3 冻结参数、种子 `20240821` 和 `mc` 方法**生成一次** 3000 条相关路径；将相同路径对象传给两个方案的评价函数。
4. 用 50% 超产折价重新核算两方案 6000 个路径利润，并分别与冻结值比较；比较均值、CVaR、分位数、成本和产销量等汇总。
5. 核对 36 行敏感性表的哈希与网格完整性；记录 `rerun=false`，避免把结构检查称为整个网格重算。

冻结经济验证容差为 0.1 元；实际最大路径利润差约 0.0000000522 元。实际金额和路径计数以 [evidence/verification.json](evidence/verification.json) 为准。

**数值相符与二进制相同分别记录。** 本次 Q3 情景数组 SHA-256 与历史指纹不同；公开报告保留 `scenario_fingerprint_matches=false`。历史冻结 CSV 的路径利润与本次结果逐行对应到上述精度，但未确认底层数组字节完全一致，也未查明历史环境与当前数学库各自对指纹差异的贡献。没有为了匹配指纹改写历史值、随机种子或科学假设。

## 源码闭包与适配边界

原 Q1/Q2/Q3 完整计算入口另见[历史优化说明](optimization.md)。公开包补齐论文支持包原有的 18 个 Python 文件及两个实际历史依赖，原科学程序不做包装改写。Q2/Q3 原流程还有历史冻结清单与候选收据先决条件，本次没有把导入检查写成完整优化成功。

`src/` 中 9 个 `q*.py` 文件逐字节复制自原论文支持包 `paper_repo/source/src/`。它们包含固定方案的类型、索引、随机参数、情景生成、经济复算和农业验证的导入闭包。为保留冻结函数，部分被导入文件仍含求解器定义；复算入口不会调用优化函数。

新增 `portable_data.py` 只负责读取规范化基础并调用原有索引构造函数。新增 `scripts/reproduce.py` 是公开包的固定方案入口；`scripts/prepare_inputs.py` 从用户本地官方附件重建并逐字段核对基础。原路径依赖的 `load_q1_data` 仍保留在冻结源码中作为历史来源函数，公开入口使用 `load_portable_data`，无需任何原项目目录。

`q2_config.py` 仍含历史 `frontier_theta` 默认字段，目的是保持冻结源码哈希；本入口不会读取该字段展开前沿搜索，研究历史中已注明该路线被放弃。

原研究的 Q1/Q2/Q3 完整优化、上界证书重建和全敏感性重跑没有作为本包成功条件，也没有在本次执行。此范围使读者能够用小型公开包复查核心比较，同时保留研究结论的实际边界。
