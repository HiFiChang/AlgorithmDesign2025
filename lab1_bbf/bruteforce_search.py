import numpy as np
from kdtree import euclidean_distance # 使用kdtree.py中定义的距离函数

def bruteforce_search(points, query_point):
    """
    暴力搜索最近邻。

    :param points: 数据点列表或numpy数组，形状 (n, d)。
    :param query_point: 查询点 (list or numpy array)。
    :return: (best_point, best_dist_sq) 最近邻点和其距离的平方。
             返回 (None, float('inf')) 如果points为空。
    """
    if points is None or len(points) == 0:
        return None, float('inf')

    # 确保points是numpy数组以便进行高效操作，尽管euclidean_distance内部会转换
    # points_array = np.array(points) # 如果不是在循环外已经转换
    # query_point_array = np.array(query_point) # 同上

    best_point_found = None
    best_dist_sq_found = float('inf')

    for point in points:
        dist_sq = euclidean_distance(query_point, point)
        if dist_sq < best_dist_sq_found:
            best_dist_sq_found = dist_sq
            best_point_found = point
            
    return best_point_found, best_dist_sq_found

# 示例用法 (可以取消注释以进行基本测试)
# if __name__ == '__main__':
#     points_data = [[2,3], [5,4], [9,6], [4,7], [8,1], [7,2], [1,1], [3,8]]
#     query = [9,2]
#     
#     print(f"Dataset has {len(points_data)} points.")
#     nn_point, nn_dist_sq = bruteforce_search(points_data, query)
#     
#     if nn_point:
#         print(f"Query point: {query}")
#         print(f"Nearest point (Brute-force): {nn_point}")
#         print(f"Squared distance: {nn_dist_sq}")
#         print(f"Distance: {np.sqrt(nn_dist_sq)}")
#     else:
#         print("Dataset was empty.")
#
#     query2 = [0,0]
#     nn_point2, nn_dist_sq2 = bruteforce_search(points_data, query2)
#     if nn_point2:
#         print(f"Query point: {query2}")
#         print(f"Nearest point (Brute-force): {nn_point2}")
#         print(f"Squared distance: {nn_dist_sq2}")
#         print(f"Distance: {np.sqrt(nn_dist_sq2)}")
#     else:
#         print("Dataset was empty for query2.") 