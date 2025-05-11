# 基于层次聚类树的图像检索系统

本项目实现了一个基于层次聚类和词袋模型的图像检索系统，能够从图像数据集中检索与查询图像相似的图像。

## 项目结构

- `feature_extractor.py`: 特征提取模块，使用SIFT提取图像特征
- `visual_dictionary.py`: 视觉词典构建模块，使用层次聚类算法
- `bow_representation.py`: 词袋表示模块，将图像转换为词频直方图
- `image_retrieval_system.py`: 图像检索系统主模块
- `run_experiments_part2.py`: 实验运行脚本
- `query_examples.py`: 查询示例脚本
- `report_part2.md`: 详细实验报告
- `README_part2.md`: 原始README文件
- `image_retrieval_results.csv`: 实验结果数据
- `vocab_size_comparison_sift.png`: 词典大小比较图
- `config.py`: 配置文件
- `photo/`: 图像数据集目录
- `features/`: 特征存储目录
- `results/`: 查询结果存储目录

## 功能特点

- SIFT特征提取：从图像中提取SIFT特征描述符
- 层次聚类：使用层次聚类算法构建视觉词典
- 词袋表示：将图像表示为视觉词频直方图
- 相似度计算：使用余弦相似度计算图像相似性
- 检索评估：提供多种性能评估指标

## 运行实验

```bash
# 运行基本实验
python run_experiments_part2.py --run-basic

# 比较不同词典大小的性能
python run_experiments_part2.py --compare-vocab-sizes

# 运行所有实验并强制重新计算
python run_experiments_part2.py --run-all --force-recompute --save-csv
```

## 查询示例

```bash
# 运行随机查询示例
python query_examples.py
```

## 实验结果

详细的实验结果和分析请参见`report_part2.md`。实验表明，该系统能够有效地检索与查询图像相似的图像，词典大小为200时性能最佳。 