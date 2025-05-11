# 图像检索系统

这个项目实现了两种图像检索系统：
1. 基于改进的SIFT特征和层次聚类树的图像检索系统
2. 多特征融合的图像检索系统（结合SIFT特征和颜色特征）

## 项目结构

```
lab2_image_retrieval/
├── core/                # 核心组件
│   ├── feature_extractor.py      # SIFT特征提取器
│   ├── color_feature_extractor.py # 颜色特征提取器
│   ├── visual_dictionary.py      # 视觉词典构建
│   ├── improved_bow.py           # 改进的词袋模型
│   └── config.py                 # 配置参数
├── systems/             # 检索系统实现
│   ├── improved_retrieval_system.py  # 改进的SIFT特征检索系统
│   └── multi_feature_retrieval_system.py  # 多特征融合检索系统
├── scripts/             # 运行脚本
│   ├── run_improved_system.py       # 运行改进的SIFT系统
│   └── run_multi_feature_system.py  # 运行多特征融合系统
├── features/            # 特征存储目录
├── improved_results/    # 改进系统结果目录
├── multi_feature_results/ # 多特征系统结果目录
├── photo/                 # 图像数据集
├── image_retrieval.py     # 主运行脚本
└── README.md              # 项目说明
```

## 系统特点

### 改进的SIFT特征检索系统
- 空间金字塔匹配保留特征的空间分布信息
- 查询自匹配优化确保查询图像排在第一位
- 词袋表示优化（TF-IDF加权，调整词典大小等）
- 多样性相似度计算方法支持

### 多特征融合检索系统
- 融合SIFT特征和颜色特征，全面表示图像内容
- 颜色特征提供全局信息，弥补SIFT特征的局限性
- 加权融合机制，灵活调整不同特征的重要性
- HSV/RGB颜色空间支持，捕获丰富的颜色分布信息

## 使用方法

### 统一运行接口

```bash
# 运行改进的SIFT特征检索系统
python image_retrieval.py --system improved --query photo/1.png

# 运行多特征融合检索系统
python image_retrieval.py --system multi --query photo/1.png --sift_weight 0.6 --color_weight 0.4

# 执行多次随机查询
python image_retrieval.py --system multi --random_queries 5
```

### 参数说明

- `--system`：选择系统类型（improved/multi）
- `--query`：指定查询图像路径
- `--random_queries`：随机执行的查询次数
- `--force_recompute`：强制重新计算特征和词典
- `--vocab_size`：SIFT词典大小
- `--sim_method`：相似度计算方法（cosine/euclidean/combined/rerank）
- `--color_bins`：颜色直方图每通道柱数（仅多特征系统）
- `--color_space`：颜色空间（rgb/hsv，仅多特征系统）
- `--sift_weight`：SIFT特征权重（仅多特征系统）
- `--color_weight`：颜色特征权重（仅多特征系统）

## 实验结果

实验表明，添加颜色特征后的多特征融合系统在检索性能上有显著提升，特别是对于那些颜色信息具有明显辨识性的图像。系统在自匹配测试中达到了100%的准确率。
