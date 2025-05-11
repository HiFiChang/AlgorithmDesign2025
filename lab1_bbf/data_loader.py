import numpy as np

def load_data_file(filepath):
    """
    加载并解析由 generateData.cpp 生成的数据文件。

    :param filepath: 数据文件的路径。
    :return: (points, queries, n, m, d)
             points: numpy数组，形状 (n, d)，包含n个数据点。
             queries: numpy数组，形状 (m, d)，包含m个查询点。
             n: 数据点的数量。
             m: 查询点的数量。
             d: 数据的维度。
    """
    with open(filepath, 'r') as f:
        # 读取第一行: n, m, d
        first_line = f.readline().strip().split()
        n = int(first_line[0])
        m = int(first_line[1])
        d = int(first_line[2])

        points_list = []
        for _ in range(n):
            line_data = list(map(int, f.readline().strip().split()))
            points_list.append(line_data)
        
        queries_list = []
        for _ in range(m):
            line_data = list(map(int, f.readline().strip().split()))
            queries_list.append(line_data)
            
    points = np.array(points_list)
    queries = np.array(queries_list)
    
    # 验证读取的数据形状是否正确
    if points.shape != (n, d):
        raise ValueError(f"Error reading points from {filepath}. Expected shape ({n}, {d}), got {points.shape}")
    if queries.shape != (m, d):
        raise ValueError(f"Error reading queries from {filepath}. Expected shape ({m}, {d}), got {queries.shape}")

    return points, queries, n, m, d

# 示例用法 (可以取消注释以进行基本测试)
# if __name__ == '__main__':
#     # 假设你的工作目录下有 data/1.txt 文件
#     # 你可能需要根据实际的 generateData.cpp 输出调整路径
#     try:
#         points, queries, n, m, d = load_data_file("data/1.txt")
#         print(f"Successfully loaded data from data/1.txt")
#         print(f"Number of data points (n): {n}, Dimensions (d): {d}")
#         print(f"Shape of points array: {points.shape}")
#         print(f"First 5 points:\n{points[:5]}")
#         print(f"Number of query points (m): {m}")
#         print(f"Shape of queries array: {queries.shape}")
#         print(f"First 3 queries:\n{queries[:3]}")
#     except FileNotFoundError:
#         print("Error: data/1.txt not found. Make sure generateData.cpp has been run and data is in the correct location.")
#     except Exception as e:
#         print(f"An error occurred: {e}") 