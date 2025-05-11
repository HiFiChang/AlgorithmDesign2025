import heapq
import numpy as np
from kdtree import KDNode, euclidean_distance # 假设 kdtree.py 在同一目录下或PYTHONPATH中

def bbf_search(root_node, query_point, t_max_leaf_nodes):
    """
    Best Bin First (BBF) 近似最近邻搜索算法。

    :param root_node: k-d树的根节点 (KDNode 类型)。
    :param query_point: 查询点 (list or numpy array)。
    :param t_max_leaf_nodes: 最大搜索的叶子节点数。
    :return: (best_point, best_dist_sq) 近似最近邻点和其距离的平方。
             返回 (None, float('inf')) 如果树为空或未找到。
    """
    if root_node is None:
        return None, float('inf')

    # Python的heapq是最小堆，BBF要求最大堆优先队列，且优先级是 1/distance_to_split。
    # 为了使用最小堆实现最大堆效果，我们将存储 (-priority, item)。
    # 初始时，可以将根节点视为一个特殊的"路径"，或者直接将其子节点加入。
    # 伪代码是 PriorityQueue ← k-d tree root with priority=0
    # 这可能意味着根节点本身或者其代表的整个空间先入队。
    # 另一种解释是，从根节点开始探索，将其子节点放入队列。
    # 我们选择将 (priority, node) 元组放入堆中。
    # 对于根节点，到其分割平面的距离可以认为是0（如果查询点恰好在平面上）
    # 或者可以先处理根节点，将其子节点加入队列。

    # (priority, node_object) - priority 越大越好 (对应 1/dist)
    # heapq 使用 (-actual_priority, item) 来模拟最大堆
    priority_queue = [] 
    
    # 初始时，我们将根节点（如果它不是叶子）的子节点加入队列
    # 或者如果根节点就是叶子，就直接处理它
    # 为了简化，我们让第一个元素是根节点本身，其优先级可以设为最高（例如，负无穷的相反数，因为它肯定要被探索）
    # 或者，更符合逻辑地，从根节点开始，并将其子节点推入队列。

    # 我们将节点本身放入队列，而不是子树。优先级计算将基于其父节点的分割。
    # (negative_priority, tie_breaker, node)
    # tie_breaker 是为了防止相同优先级的节点比较时出错（如果节点本身不可比较）
    # 这里用一个计数器作为 tie_breaker
    counter = 0
    heapq.heappush(priority_queue, (0, counter, root_node)) # 初始根节点优先级最高 (用0代表-infinity for max_heap)
    counter += 1

    best_point_found = None
    best_dist_sq_found = float('inf')
    searched_leaf_nodes_count = 0

    while priority_queue and searched_leaf_nodes_count < t_max_leaf_nodes:
        neg_priority, _, current_node = heapq.heappop(priority_queue)
        # priority = -neg_priority

        if current_node is None: # Should not happen if we push valid nodes
            continue

        if current_node.is_leaf:
            searched_leaf_nodes_count += 1
            for point_in_leaf in current_node.points_in_leaf:
                dist_sq = euclidean_distance(query_point, point_in_leaf)
                if dist_sq < best_dist_sq_found:
                    best_dist_sq_found = dist_sq
                    best_point_found = point_in_leaf
            # 一旦叶子节点处理完毕，继续下一个更高优先级的节点
            continue 
        
        # 非叶节点 (current_node.point 是分割点, current_node.split_dim 是分割维度)
        split_dim = current_node.split_dim
        pivot_point = current_node.point
        query_coord_at_split_dim = query_point[split_dim]
        pivot_coord_at_split_dim = pivot_point[split_dim]

        # 决定"更近"和"更远"的子树
        if query_coord_at_split_dim < pivot_coord_at_split_dim:
            nearer_child = current_node.left
            farther_child = current_node.right
        else:
            nearer_child = current_node.right
            farther_child = current_node.left

        # 将子节点加入优先队列
        # 优先级是 1 / (到分割超平面的距离)
        # 如果距离为0，优先级为无穷大。我们用一个很大的数代替。
        # 注意：我们总是探索更近的子节点，但两者都以其各自的优先级入队。
        # 伪代码是：
        # queue.insert(farther_subtree, priority=1/(distance_to_split))
        # queue.insert(nearer_subtree, priority=1/(distance_to_split))
        # 这里的 distance_to_split 应该是从 query_point 到 *父节点* 的分割超平面的距离。
        # 当我们将子节点放入队列时，这个"父节点的分割超平面"就是当前 current_node 的超平面。
        
        dist_to_split_plane_sq = (query_coord_at_split_dim - pivot_coord_at_split_dim) ** 2
        # 我们需要距离，而不是距离的平方，用于 1/dist
        dist_to_split_plane = np.sqrt(dist_to_split_plane_sq)

        # 优先级计算：1 / (distance_to_hyperplane_of_parent)
        # 如果 distance is 0, priority is effectively infinity.
        # To avoid division by zero, if dist_to_split_plane is very small, use a large priority.
        priority_val = 1.0 / (dist_to_split_plane + 1e-9) # Add epsilon to avoid division by zero
        
        if nearer_child is not None:
            # 对于更近的子节点，它被优先探索。
            # 它的优先级也可以基于其父分割（即当前节点）计算。
            heapq.heappush(priority_queue, (-priority_val, counter, nearer_child))
            counter +=1
            
        if farther_child is not None:
            # 对于更远的子节点，其优先级也基于其父分割（即当前节点）计算。
            heapq.heappush(priority_queue, (-priority_val, counter, farther_child))
            counter += 1
            
        # BBF 也应该考虑当前分割点本身是否是最近邻
        # 这个检查在叶节点处理或者标准k-d树中有，BBF这里也应该考虑
        # 根据伪代码，只在叶节点更新 best_dist。但通常分割点也应被检查。
        # 为遵循伪代码，此处不检查 current_node.point，只在叶节点检查。
        # 如果要求更精确，或者标准BBF变体，此处可以加入检查：
        # dist_to_pivot_sq = euclidean_distance(query_point, pivot_point)
        # if dist_to_pivot_sq < best_dist_sq_found:
        #    best_dist_sq_found = dist_to_pivot_sq
        #    best_point_found = pivot_point

    return best_point_found, best_dist_sq_found

# 示例用法 (可以取消注释以进行基本测试)
# if __name__ == '__main__':
#     from kdtree import build_kdtree # 需要 kdtree.py 中的 build_kdtree
#     points_data = [[2,3], [5,4], [9,6], [4,7], [8,1], [7,2], [1,1], [3,8]]
#     kdtree_root = build_kdtree(points_data, leaf_max_size=2)

#     if kdtree_root:
#         query = [9,2]
#         t_leaves = 3 # 搜索最多3个叶子节点
#         
#         print(f"Building k-d tree with {len(points_data)} points and leaf_max_size=2")
#         
#         # 为了能运行，确保kdtree.py中KDNode和euclidean_distance可用
#         # 且build_kdtree能正常工作
#         try:
#             print("Running BBF Search...")
#             approx_nn, approx_dist_sq = bbf_search(kdtree_root, query, t_leaves)
#             if approx_nn:
#                 print(f"Query point: {query}")
#                 print(f"Approximate Nearest point (BBF, t={t_leaves}): {approx_nn}")
#                 print(f"Approximate Squared distance: {approx_dist_sq}")
#                 print(f"Approximate Distance: {np.sqrt(approx_dist_sq)}")
#             else:
#                 print("BBF search did not find a point or tree was empty.")
#         except Exception as e:
#             print(f"An error occurred during BBF search: {e}")
#             import traceback
#             traceback.print_exc()
            
#         # 对比标准k-d树搜索 (如果已实现且可用)
#         # from kdtree import search_kdtree
#         # exact_nn, exact_dist_sq = search_kdtree(kdtree_root, query)
#         # if exact_nn:
#         #     print(f"Exact Nearest point (k-d tree): {exact_nn}")
#         #     print(f"Exact Squared distance: {exact_dist_sq}")
#     else:
#         print("Failed to build k-d tree.") 