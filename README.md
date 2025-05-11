# 算法实验项目

本项目包含两个独立的算法实验：

1. **BBF算法实现与对比**：实现和评估基于k-d树的Best Bin First (BBF)近似最近邻搜索算法
2. **基于层次聚类树的图像检索系统**：实现基于层次聚类和词袋模型的图像检索系统

## 项目结构

- `lab1_bbf/`: BBF算法实验相关代码和文档
- `lab2_image_retrieval/`: 图像检索系统相关代码和文档

## 实验一：BBF算法实现与对比

本实验实现了k-d树构建算法和BBF搜索算法，并在不同维度的数据集上进行了对比实验，包括暴力搜索、标准k-d树搜索和BBF搜索，分析了它们在查询时间、准确率和内存占用方面的表现。

详细内容请参见[BBF算法实验README](lab1_bbf/README.md)。

## 实验二：基于层次聚类树的图像检索系统

本实验实现了一个完整的图像检索系统，包括SIFT特征提取、层次聚类构建视觉词典、词袋表示和图像查询，能够从图像数据集中检索与查询图像相似的图像。

详细内容请参见[图像检索系统README](lab2_image_retrieval/README.md)。

## 运行环境

- Python 3.6+
- 依赖库：numpy, scipy, scikit-learn, opencv-python, matplotlib, tqdm 