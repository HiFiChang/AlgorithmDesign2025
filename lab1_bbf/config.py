# config.py

# BBF 算法参数
BBF_T_VALUE = 200  # BBF搜索时检查的最大叶子节点数 (t=200)

# k-d 树构建参数
LEAF_MAX_SIZE = 10 # k-d树叶子节点中允许的最大点数
                    # 这个值可以根据数据集大小和特性进行调整
                    # 较小的值可能导致树更深，但叶子更纯
                    # 较大的值树较浅，但叶子内点多，可能增加叶内搜索时间

# 实验参数
# 使用新的文件命名结构：data_<dim>D_<num>.txt
DATA_FILE_PATH_TEMPLATE = "data/data_{dimension}D_{num}.txt" # 数据文件路径模板
DIMENSIONS_TO_TEST = [2, 4, 8, 16]  # 要测试的不同维度
FILES_PER_DIMENSION = 5   # 每个维度的数据文件数量

# 如果要沿用旧的数据文件命名方式，可以将下面的变量设为True，并设置NUM_DATA_FILES_TO_PROCESS
USE_OLD_DATA_FILES = False
OLD_DATA_FILE_PATH_TEMPLATE = "data/{}.txt"
NUM_DATA_FILES_TO_PROCESS = 10  # 处理多少个旧数据文件 (1 到 100)。
                              # 最终实验应在多个文件上运行并平均结果。

# 准确率定义
ACCURACY_THRESHOLD = 1.05 # 返回结果与真实最近邻的欧氏距离比值（≤1.05视为成功）

# 实验控制
SKIP_BRUTEFORCE_FOR_HIGH_DIM = True  # 对于高维度（>8），是否跳过暴力搜索（可能非常慢）
HIGH_DIM_THRESHOLD = 8      # 何为"高维度"的阈值
VERBOSE_OUTPUT = True       # 是否输出详细日志
SAVE_RESULTS_TO_CSV = True  # 是否将结果保存到CSV文件
CSV_RESULT_PATH = "bbf_experiment_results.csv"  # CSV文件路径

# 其他
# DATA_DIMENSION = 8 # 从 generateData.cpp 可知是8维，也可以从数据文件动态读取 