import heapq
import numpy as np

class KDNode:
    """k-d树的节点"""
    def __init__(self, point=None, split_dim=None, left=None, right=None, is_leaf=False, points_in_leaf=None):
        self.point = point  # 分割点 (仅非叶节点)
        self.split_dim = split_dim  # 分割维度 (仅非叶节点)
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.is_leaf = is_leaf # 是否为叶节点
        self.points_in_leaf = points_in_leaf if points_in_leaf is not None else [] # 叶节点包含的点列表

def euclidean_distance(point1, point2):
    """计算两个点之间的欧氏距离的平方"""
    return np.sum((np.array(point1) - np.array(point2))**2)

def build_kdtree(points, depth=0, leaf_max_size=10):
    """
    构建k-d树
    :param points: 数据点列表，每个点是一个列表或numpy数组
    :param depth: 当前树的深度，用于决定分割维度
    :param leaf_max_size: 叶子节点能容纳的最大数据点数量
    :return: k-d树的根节点
    """
    if not points:
        return None

    n_points = len(points)
    if n_points == 0:
        return None

    # 检查points中的元素是否都是可迭代的
    if not all(hasattr(p, '__iter__') for p in points):
        raise ValueError("Points list must contain iterable points.")
        
    # 尝试获取维度，假设所有点维度相同
    try:
        k = len(points[0]) 
    except TypeError: # points[0] 不是可迭代的
        raise ValueError("Points must be iterable and contain coordinates.")


    if n_points <= leaf_max_size:
        return KDNode(is_leaf=True, points_in_leaf=list(points))

    # 选择分割维度
    split_dim = depth % k

    # 根据分割维度对点进行排序，并选择中位数作为分割点
    # 注意：为了避免直接修改原始points列表的副本，这里转换为numpy数组进行操作
    # 或者使用 sorted(points, key=lambda x: x[split_dim]) 如果 points 是列表的列表
    try:
        # 确保 points 是一个 numpy 数组或者可以转换为 numpy 数组的结构
        points_array = np.array(points)
        if points_array.ndim == 1 and k > 0 : # e.g. [[1,2,3]] will be array([1,2,3]), we want array([[1,2,3]])
             points_array = np.array([points])
        elif points_array.ndim != 2:
             raise ValueError(f"Points should be a 2D array-like structure, got shape {points_array.shape}")
        
        # Sort points based on the split_dim dimension
        # argsort returns indices that would sort the array
        sorted_indices = points_array[:, split_dim].argsort()
        points_sorted = points_array[sorted_indices]
        
    except IndexError:
        # This can happen if points are not uniform in dimension or k is incorrect
        raise ValueError(f"Error accessing dimension {split_dim} for points. Ensure all points have at least {k} dimensions.")


    median_idx = n_points // 2
    median_point = points_sorted[median_idx]

    # 创建节点
    node = KDNode(point=median_point.tolist(), split_dim=split_dim) # Store median_point as list

    # 递归构建左右子树
    # points_sorted已经是numpy数组，可以直接切片
    node.left = build_kdtree(points_sorted[:median_idx].tolist(), depth + 1, leaf_max_size)
    node.right = build_kdtree(points_sorted[median_idx + 1:].tolist(), depth + 1, leaf_max_size)
    
    # 如果一个子树为空，但中位数本身应该在父节点，叶子节点应包含数据
    # 当前实现中，中位数点仅作为分割点，不直接存储在叶子中，除非它落入叶子大小条件
    # 如果中位数点也需要被包含，需要调整逻辑

    return node

def _kdtree_search_recursive(node, query_point, best_dist_sq, best_node, depth, k_dim):
    """
    k-d树最近邻搜索的递归辅助函数。
    返回 (最近点, 最近距离的平方)
    """
    if node is None:
        return best_node, best_dist_sq

    current_best_node = best_node
    current_best_dist_sq = best_dist_sq

    if node.is_leaf:
        for point_in_leaf in node.points_in_leaf:
            dist_sq = euclidean_distance(query_point, point_in_leaf)
            if dist_sq < current_best_dist_sq:
                current_best_dist_sq = dist_sq
                current_best_node = point_in_leaf
        return current_best_node, current_best_dist_sq

    # 非叶节点
    split_dim = node.split_dim
    node_point = node.point # 分割点

    # 决定先访问哪个子树
    if query_point[split_dim] < node_point[split_dim]:
        nearer_subtree = node.left
        farther_subtree = node.right
    else:
        nearer_subtree = node.right
        farther_subtree = node.left

    # 递归搜索更近的子树
    current_best_node, current_best_dist_sq = _kdtree_search_recursive(
        nearer_subtree, query_point, current_best_dist_sq, current_best_node, depth + 1, k_dim
    )

    # 检查是否需要搜索更远的子树
    # (查询点到分割轴的距离)^2
    dist_to_split_sq = (query_point[split_dim] - node_point[split_dim]) ** 2
    if dist_to_split_sq < current_best_dist_sq:
        # 如果以查询点为圆心，当前最优距离为半径的圆与分割超平面相交，
        # 则需要在另一侧子树中继续搜索
        current_best_node, current_best_dist_sq = _kdtree_search_recursive(
            farther_subtree, query_point, current_best_dist_sq, current_best_node, depth + 1, k_dim
        )
    
    # 也要检查当前分割点是否是更近的点
    # (在我们的实现中，分割点不直接存储在叶子中，但它本身也是一个数据点，只是用于分割)
    # 标准的k-d树实现中，非叶子节点的point也参与比较
    dist_to_node_point_sq = euclidean_distance(query_point, node_point)
    if dist_to_node_point_sq < current_best_dist_sq:
        current_best_dist_sq = dist_to_node_point_sq
        current_best_node = node_point

    return current_best_node, current_best_dist_sq

def search_kdtree(root, query_point):
    """
    在k-d树中搜索最近邻。
    :param root: k-d树的根节点。
    :param query_point: 查询点。
    :return: (最近点, 最近距离的平方) 如果找到，否则 (None, float('inf'))
    """
    if root is None:
        return None, float('inf')
    
    # 假设点的维度可以从根节点的一个点或者子节点中获取，或者作为参数传入
    # 这里我们假设 query_point 和树中点的维度一致
    try:
        k_dim = len(query_point)
    except TypeError:
        raise ValueError("Query point must be an iterable representing coordinates.")

    return _kdtree_search_recursive(root, query_point, float('inf'), None, 0, k_dim)


# 示例用法 (可以取消注释以进行基本测试)
# if __name__ == '__main__':
#     points = [[2,3], [5,4], [9,6], [4,7], [8,1], [7,2]]
#     root = build_kdtree(points, leaf_max_size=2)
# 
#     query = [9,2]
#     nearest_node, nearest_dist_sq = search_kdtree(root, query)
#     if nearest_node:
#         print(f"Query point: {query}")
#         print(f"Nearest point: {nearest_node}")
#         print(f"Squared distance: {nearest_dist_sq}")
#         print(f"Distance: {np.sqrt(nearest_dist_sq)}")
#     else:
#         print("Tree is empty or search failed.")
# 
#     query2 = [1,1]
#     nearest_node2, nearest_dist_sq2 = search_kdtree(root, query2)
#     if nearest_node2:
#         print(f"Query point: {query2}")
#         print(f"Nearest point: {nearest_node2}")
#         print(f"Squared distance: {nearest_dist_sq2}")
#         print(f"Distance: {np.sqrt(nearest_dist_sq2)}")
#     else:
#         print("Tree is empty or search failed for query2.") 