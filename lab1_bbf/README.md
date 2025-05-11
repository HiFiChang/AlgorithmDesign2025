# BBF算法实现与对比

本项目实现了基于k-d树的Best Bin First (BBF)近似最近邻搜索算法，并与暴力搜索和标准k-d树搜索进行了对比分析。

## 项目结构

- `kdtree.py`: k-d树的实现
- `bbf_search.py`: BBF算法实现
- `bruteforce_search.py`: 暴力搜索实现
- `data_loader.py`: 数据加载工具
- `run_experiments_part1.py`: 实验运行脚本
- `report_part1.md`: 详细实验报告
- `generate_all_data.py`: 数据集生成脚本
- `generateData.cpp`: C++版数据生成程序
- `bbf_experiment_results.csv`: 实验结果数据
- `data/`: 实验数据集目录

## 实验内容

1. 实现k-d树构建算法，支持欧氏距离计算
2. 实现BBF搜索算法，使用最大堆优先队列管理搜索路径
3. 在不同维度的数据集上验证算法正确性
4. 对比暴力搜索、标准k-d树搜索和BBF搜索的性能

## 运行实验

```bash
python run_experiments_part1.py
```

## 实验结果

详细的实验结果和分析请参见`report_part1.md`。

实验表明BBF算法在查询时间和准确率之间取得了平衡，特别是在高维空间中能够保持较高的查询效率，但随着维度增加准确率会有所下降。 