import time
import numpy as np
import os
import csv
import argparse
import sys
import tracemalloc  # 导入tracemalloc库用于测量内存占用
from datetime import datetime

# 从我们创建的模块导入
import config
from data_loader import load_data_file
from kdtree import build_kdtree, search_kdtree, euclidean_distance
from bbf_search import bbf_search
from bruteforce_search import bruteforce_search

def run_single_experiment(data_points, query_points, dimension, skip_bruteforce=False):
    """对一组数据点和查询点运行所有搜索算法并收集结果。"""
    results = {
        'bruteforce': {'times': [], 'distances_sq': [], 'points': [], 'avg_time': 0, 'accuracy_points': [], 'accuracy_distances_sq': [], 'memory_mb': 0},
        'kdtree_exact': {'times': [], 'distances_sq': [], 'points': [], 'avg_time': 0, 'accuracy_points': [], 'accuracy_distances_sq': [], 'memory_mb': 0},
        'bbf': {'times': [], 'distances_sq': [], 'points': [], 'avg_time': 0, 'accuracy_points': [], 'accuracy_distances_sq': [], 'memory_mb': 0}
    }

    # 1. 构建 k-d 树
    if config.VERBOSE_OUTPUT:
        print(f"Building k-d tree with {len(data_points)} points and leaf_max_size={config.LEAF_MAX_SIZE}...")
    
    tracemalloc.start()
    build_start_time = time.time()
    # 将data_points转换为Python列表以供build_kdtree使用，这部分内存会计入峰值
    data_points_list_for_kdtree = data_points.tolist()
    kdtree_root = build_kdtree(data_points_list_for_kdtree, leaf_max_size=config.LEAF_MAX_SIZE) 
    current_kdtree, peak_kdtree = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    kdtree_memory_mb = peak_kdtree / (1024 * 1024)
    results['kdtree_exact']['memory_mb'] = kdtree_memory_mb
    results['bbf']['memory_mb'] = kdtree_memory_mb
    
    build_time = time.time() - build_start_time # Corrected build_time calculation
    if config.VERBOSE_OUTPUT:
        print(f"K-d tree built in {build_time:.4f} seconds.")
        print(f"K-d tree peak memory usage (tracemalloc): {kdtree_memory_mb:.2f} MB")

    if kdtree_root is None:
        print("Failed to build k-d tree. Skipping queries for this dataset.")
        return None

    num_queries = len(query_points)
    if config.VERBOSE_OUTPUT:
        print(f"Running {num_queries} queries...")

    # 对于暴力搜索，报告其操作的数据结构 (data_points numpy array) 的大小
    if not skip_bruteforce:
        bruteforce_data_memory_mb = data_points.nbytes / (1024 * 1024)
        results['bruteforce']['memory_mb'] = bruteforce_data_memory_mb
        if config.VERBOSE_OUTPUT:
            # 仍然可以打印操作峰值，以了解函数本身的开销
            tracemalloc.start()
            _ = bruteforce_search(data_points, query_points[0].tolist())
            _, peak_bf_op = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            print(f"Bruteforce search data memory: {bruteforce_data_memory_mb:.2f} MB")
            print(f"Bruteforce search operational peak memory (tracemalloc): {peak_bf_op / (1024*1024):.2f} MB")

    # 测量BBF搜索的优先队列等额外动态内存开销
    # 我们只在第一次查询时测量这个额外开销，假设它具有代表性
    if num_queries > 0:
        tracemalloc.start()
        _ = bbf_search(kdtree_root, query_points[0].tolist(), config.BBF_T_VALUE)
        _, peak_bbf_op = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        bbf_extra_op_memory_mb = peak_bbf_op / (1024 * 1024)
        if config.VERBOSE_OUTPUT:
            print(f"BBF search operational peak memory (tracemalloc for priority queue etc.): {bbf_extra_op_memory_mb:.2f} MB")
        # 将这个操作峰值加到BBF的内存上，如果它比已有的k-d树内存大（不太可能，但作为一种保护）
        # 或者，更准确地说，我们假设这个操作峰值是*除了*k-d树之外的额外峰值。
        # 然而，tracemalloc的peak是指整个跟踪会话中的峰值。
        # 为了简化，我们只把k-d树的内存作为bbf的基础，BBF的额外动态分配如果很少，那它的总内存就约等于k-d树。
        # 如果需要更精确分离，会更复杂。
        # 鉴于BBF额外内存很小，当前 `results['bbf']['memory_mb']` 已设为k-d树内存，这在多数情况下是主要部分。
        # 如果要体现那一点点额外开销，需要小心避免重复计算。
        # 一个简单的做法是: 假设bbf_search中的动态分配是其主要额外开销，并且独立于k-d树的初始分配。
        # results['bbf']['memory_mb'] = kdtree_memory_mb + bbf_extra_op_memory_mb 
        # 但是，因为kdtree_memory_mb已经是peak_kdtree，它已经包含了Python列表转换的内存，
        # 而bbf_search也可能在其内部做一些小列表转换。为了避免过度复杂化和潜在高估，
        # 保持BBF的额外内存测量为小值，并主要关注k-d树的内存。
        # 如果那0.01MB的额外开销很重要，可以加，但要确保解释清晰。
        # 考虑到之前的讨论和结果，BBF的内存主要是其依赖的K-D树的内存，额外队列开销很小。
        # 所以，`results['bbf']['memory_mb'] = kdtree_memory_mb` 是合理的基线。
        # 如果想明确加上测量的额外BBF操作内存:
        # results['bbf']['memory_mb'] = kdtree_memory_mb + bbf_extra_op_memory_mb 
        # 这会导致BBF内存略高于k-d树，如之前观察到的那样，这是OK的。
        results['bbf']['memory_mb'] = kdtree_memory_mb + bbf_extra_op_memory_mb

    for i, q_point in enumerate(query_points):
        if config.VERBOSE_OUTPUT and ((i + 1) % 10 == 0 or i == num_queries - 1): # Print progress
            print(f"  Processing query {i+1}/{num_queries}...")
        
        q_point_list = q_point.tolist() # Ensure query point is in list format if needed by functions

        # a) 暴力搜索
        if not skip_bruteforce:
            start_time = time.perf_counter()
            bf_nn, bf_dist_sq = bruteforce_search(data_points, q_point_list)
            end_time = time.perf_counter()
            results['bruteforce']['times'].append(end_time - start_time)
            results['bruteforce']['distances_sq'].append(bf_dist_sq)
            results['bruteforce']['points'].append(bf_nn)
        else:
            pass

        # b) 标准 k-d 树搜索
        start_time = time.perf_counter()
        kd_nn, kd_dist_sq = search_kdtree(kdtree_root, q_point_list)
        end_time = time.perf_counter()
        results['kdtree_exact']['times'].append(end_time - start_time)
        results['kdtree_exact']['distances_sq'].append(kd_dist_sq)
        results['kdtree_exact']['points'].append(kd_nn)

        # c) BBF 搜索
        start_time = time.perf_counter()
        bbf_nn, bbf_dist_sq = bbf_search(kdtree_root, q_point_list, config.BBF_T_VALUE)
        end_time = time.perf_counter()
        results['bbf']['times'].append(end_time - start_time)
        results['bbf']['distances_sq'].append(bbf_dist_sq)
        results['bbf']['points'].append(bbf_nn)

    # 如果跳过了暴力搜索，用标准k-d树的结果作为基准
    if skip_bruteforce:
        results['bruteforce'] = results['kdtree_exact'].copy()
        
    # 计算平均时间
    for algo in results:
        if results[algo]['times']:
            results[algo]['avg_time'] = np.mean(results[algo]['times'])

    # 计算准确率 (与暴力搜索结果对比)
    # 准确率：返回结果与真实最近邻的欧氏距离比值（≤1.05视为成功）
    # 我们用暴力搜索或标准k-d树的结果作为"真实"最近邻
    true_distances_sq = results['bruteforce']['distances_sq']
    
    for algo in ['kdtree_exact', 'bbf']:
        successful_queries = 0
        algo_distances_sq = results[algo]['distances_sq']
        if not algo_distances_sq or not true_distances_sq or len(algo_distances_sq) != len(true_distances_sq):
            results[algo]['accuracy'] = 0.0
            continue
            
        for idx in range(len(algo_distances_sq)):
            # Ground truth distance might be zero if query point is identical to a data point
            # Algorithm distance might also be zero.
            # If true_dist_sq is 0, algo_dist_sq must also be 0 for it to be a "success" in ratio terms.
            # (or very close, due to float precision)
            
            true_dist = np.sqrt(true_distances_sq[idx])
            algo_dist = np.sqrt(algo_distances_sq[idx])
            
            results[algo]['accuracy_points'].append(results[algo]['points'][idx])
            results[algo]['accuracy_distances_sq'].append(algo_distances_sq[idx])

            if true_dist == 0: # Query point is one of the data points
                if algo_dist == 0: # Algorithm found the exact point
                    successful_queries += 1
                # else: if algo_dist > 0, it's a miss, ratio would be undefined or infinite.
            elif algo_dist / true_dist <= config.ACCURACY_THRESHOLD:
                successful_queries += 1
        
        results[algo]['accuracy'] = (successful_queries / num_queries) * 100 if num_queries > 0 else 0

    # 添加一些额外的元数据到结果中
    results['metadata'] = {
        'dimension': dimension,
        'data_points': len(data_points),
        'query_points': num_queries,
        'kdtree_build_time': build_time,
        'leaf_max_size': config.LEAF_MAX_SIZE,
        'bbf_t_value': config.BBF_T_VALUE
    }

    return results

def save_results_to_csv(all_results, filepath):
    """将实验结果保存到CSV文件中"""
    # 检查是否存在现有文件，确定是否需要写入标题行
    write_header = not os.path.exists(filepath)
    
    with open(filepath, 'a', newline='') as f:
        fieldnames = [
            'datetime', 'dimension', 'data_points', 'query_points', 'leaf_max_size', 'bbf_t_value',
            'kdtree_build_time', 
            'bruteforce_avg_time', 'kdtree_avg_time', 'bbf_avg_time',
            'kdtree_accuracy', 'bbf_accuracy',
            'bruteforce_memory_mb', 'kdtree_memory_mb', 'bbf_memory_mb' # 确保这里的列名与之后writerow的键一致
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()
        
        # 确保 avg_row 中的键与 fieldnames 完全对应
        for dim_key, dim_results_dict in all_results.items(): # 使用不同的变量名以区分
            # 对于每个维度，汇总所有文件的平均结果
            # 从 dim_results_dict 中提取各算法的结果
            bruteforce_res = dim_results_dict.get('bruteforce', {})
            kdtree_res = dim_results_dict.get('kdtree_exact', {})
            bbf_res = dim_results_dict.get('bbf', {})
            metadata_res = dim_results_dict.get('metadata', {})

            avg_row = {
                'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'dimension': metadata_res.get('dimension', dim_key),
                'data_points': metadata_res.get('data_points', 0),
                'query_points': metadata_res.get('query_points', 0),
                'leaf_max_size': metadata_res.get('leaf_max_size', config.LEAF_MAX_SIZE),
                'bbf_t_value': metadata_res.get('bbf_t_value', config.BBF_T_VALUE),
                'kdtree_build_time': metadata_res.get('kdtree_build_time', 0),
                'bruteforce_avg_time': bruteforce_res.get('avg_time', 0),
                'kdtree_avg_time': kdtree_res.get('avg_time', 0),
                'bbf_avg_time': bbf_res.get('avg_time', 0),
                'kdtree_accuracy': kdtree_res.get('accuracy', 0),
                'bbf_accuracy': bbf_res.get('accuracy', 0),
                'bruteforce_memory_mb': bruteforce_res.get('memory_mb', 0),
                'kdtree_memory_mb': kdtree_res.get('memory_mb', 0),
                'bbf_memory_mb': bbf_res.get('memory_mb', 0)
            }
            writer.writerow(avg_row)

def test_dimension(dimension, files_per_dim):
    """对指定维度运行实验，处理多个数据文件，并返回聚合结果"""
    aggregated_results = {
        'bruteforce': {'avg_times': [], 'accuracies': [], 'memory_mbs': []},
        'kdtree_exact': {'avg_times': [], 'accuracies': [], 'memory_mbs': []},
        'bbf': {'avg_times': [], 'accuracies': [], 'memory_mbs': []},
        'metadata': {
            'dimension': dimension,
            'data_points': 0,  # 将在处理第一个文件时填充
            'query_points': 0, # 将在处理第一个文件时填充
            'kdtree_build_time': 0.0,
            'leaf_max_size': config.LEAF_MAX_SIZE,
            'bbf_t_value': config.BBF_T_VALUE
        }
    }
    
    # 是否因为高维而跳过暴力搜索
    skip_bruteforce = config.SKIP_BRUTEFORCE_FOR_HIGH_DIM and dimension > config.HIGH_DIM_THRESHOLD
    if skip_bruteforce:
        print(f"Dimension {dimension} > {config.HIGH_DIM_THRESHOLD}, skipping bruteforce search (will use k-d tree as ground truth)")
    
    total_files_processed = 0
    total_build_time = 0.0
    
    for file_num in range(1, files_per_dim + 1):
        filepath = config.DATA_FILE_PATH_TEMPLATE.format(dimension=dimension, num=file_num)
        print(f"\nProcessing data file: {filepath}...")
        if not os.path.exists(filepath):
            print(f"File {filepath} not found. Skipping.")
            continue
        
        try:
            points, queries, n, m, d = load_data_file(filepath)
            print(f"Loaded {n} data points and {m} query points, each with {d} dimensions.")
            
            # 验证维度是否匹配预期
            if d != dimension:
                print(f"Warning: File {filepath} has dimension {d}, expected {dimension}.")
                
            # 填充元数据（假设所有文件的n和m相同）
            if aggregated_results['metadata']['data_points'] == 0:
                aggregated_results['metadata']['data_points'] = n
                aggregated_results['metadata']['query_points'] = m
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            continue

        # 运行单个实验
        experiment_results = run_single_experiment(points, queries, dimension, skip_bruteforce)
        
        if experiment_results:
            print("\nResults for current file:")
            for algo_name, res_data in experiment_results.items():
                if algo_name == 'metadata':
                    total_build_time += res_data['kdtree_build_time']
                    continue
                    
                print(f"  Algorithm: {algo_name}")
                print(f"    Average Query Time: {res_data.get('avg_time', 0):.6f} seconds")
                print(f"    Memory Usage: {res_data.get('memory_mb', 0):.2f} MB")  # 显示内存占用
                if algo_name != 'bruteforce': # Bruteforce is the baseline for accuracy
                    print(f"    Accuracy: {res_data.get('accuracy', 0):.2f}%")
                
                # Store for overall aggregation
                if algo_name in aggregated_results:
                    aggregated_results[algo_name]['avg_times'].append(res_data.get('avg_time', 0))
                    aggregated_results[algo_name]['memory_mbs'].append(res_data.get('memory_mb', 0))  # 存储内存使用
                    if algo_name != 'bruteforce':
                        aggregated_results[algo_name]['accuracies'].append(res_data.get('accuracy', 0))
                        
            total_files_processed += 1
    
    # 计算平均值（只有在至少处理了一个文件的情况下）
    if total_files_processed > 0:
        # 更新元数据中的构建时间（平均值）
        aggregated_results['metadata']['kdtree_build_time'] = total_build_time / total_files_processed
        
        # 计算各算法的平均时间和准确率
        for algo_name, data in aggregated_results.items():
            if algo_name == 'metadata':
                continue
                
            if data['avg_times']:
                data['avg_time'] = np.mean(data['avg_times'])
            else:
                data['avg_time'] = 0.0
                
            if algo_name != 'bruteforce' and data['accuracies']:
                data['accuracy'] = np.mean(data['accuracies'])
            else:
                data['accuracy'] = 100.0 if algo_name == 'bruteforce' else 0.0
                
            # 计算平均内存占用
            if data['memory_mbs']:
                data['memory_mb'] = np.mean(data['memory_mbs'])
            else:
                data['memory_mb'] = 0.0
    else:
        print(f"Warning: No valid files processed for dimension {dimension}")
        
    return aggregated_results

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Run BBF algorithm comparison experiments.')
    parser.add_argument('--dimensions', type=int, nargs='+', 
                        help='List of dimensions to test (default: from config.py)')
    parser.add_argument('--files-per-dim', type=int, 
                        help=f'Number of files to process per dimension (default: {config.FILES_PER_DIMENSION})')
    parser.add_argument('--bbf-t', type=int, 
                        help=f'BBF t parameter (max leaf nodes to check) (default: {config.BBF_T_VALUE})')
    parser.add_argument('--leaf-max-size', type=int, 
                        help=f'k-d tree leaf max size (default: {config.LEAF_MAX_SIZE})')
    parser.add_argument('--skip-bruteforce', action='store_true', 
                        help='Skip bruteforce search for all dimensions')
    parser.add_argument('--use-old-data', action='store_true', 
                        help='Use old data file naming format (data/N.txt)')
    parser.add_argument('--old-files-count', type=int, 
                        help=f'Number of old format files to process (default: {config.NUM_DATA_FILES_TO_PROCESS})')
    parser.add_argument('--no-csv', action='store_true', 
                        help='Do not save results to CSV file')
    parser.add_argument('--quiet', action='store_true', 
                        help='Reduce verbosity of output')
    
    args = parser.parse_args()
    
    # 应用命令行参数到配置
    if args.dimensions:
        dimensions_to_test = args.dimensions
    else:
        dimensions_to_test = config.DIMENSIONS_TO_TEST
        
    files_per_dim = args.files_per_dim if args.files_per_dim else config.FILES_PER_DIMENSION
    
    if args.bbf_t:
        config.BBF_T_VALUE = args.bbf_t
        
    if args.leaf_max_size:
        config.LEAF_MAX_SIZE = args.leaf_max_size
        
    if args.skip_bruteforce:
        config.SKIP_BRUTEFORCE_FOR_HIGH_DIM = True
        config.HIGH_DIM_THRESHOLD = 0  # 跳过所有维度的暴力搜索
        
    if args.use_old_data:
        config.USE_OLD_DATA_FILES = True
        
    if args.old_files_count:
        config.NUM_DATA_FILES_TO_PROCESS = args.old_files_count
        
    if args.no_csv:
        config.SAVE_RESULTS_TO_CSV = False
        
    if args.quiet:
        config.VERBOSE_OUTPUT = False
    
    print("Starting Part 1: BBF Algorithm Comparison Experiments")
    print(f"Configuration: BBF_T_VALUE={config.BBF_T_VALUE}, LEAF_MAX_SIZE={config.LEAF_MAX_SIZE}")
    
    all_dimension_results = {}

    if config.USE_OLD_DATA_FILES:
        # 使用旧的文件命名格式 (data/N.txt)
        print(f"Using old data file format. Processing {config.NUM_DATA_FILES_TO_PROCESS} files.")
        
        # 这里需要额外实现旧格式文件的处理逻辑
        # ...

    else:
        # 使用新的文件命名格式 (data/data_<dim>D_<num>.txt)
        for d in dimensions_to_test:
            print(f"\n{'='*50}")
            print(f"TESTING DIMENSION: {d}")
            print(f"{'='*50}")
            
            results = test_dimension(d, files_per_dim)
            
            if results:
                all_dimension_results[d] = results
                
                print(f"\nAGGREGATED RESULTS FOR DIMENSION {d} (across {files_per_dim} files):")
                print(f"Data points: {results['metadata']['data_points']}, Query points: {results['metadata']['query_points']}")
                print(f"Average k-d tree build time: {results['metadata']['kdtree_build_time']:.4f} seconds")
                
                print("\nAlgorithm performance:")
                print(f"{'Algorithm':<15} | {'Avg. Query Time (s)':<20} | {'Accuracy (%)':<12} | {'Memory Usage (MB)':<15}")
                print(f"{'-'*15:<15} | {'-'*20:<20} | {'-'*12:<12} | {'-'*15:<15}")
                print(f"{'Bruteforce':<15} | {results['bruteforce']['avg_time']:<20.6f} | {'100.00':<12} | {results['bruteforce']['memory_mb']:.2f}")
                print(f"{'Standard k-d':<15} | {results['kdtree_exact']['avg_time']:<20.6f} | {results['kdtree_exact']['accuracy']:<12.2f} | {results['kdtree_exact']['memory_mb']:.2f}")
                print(f"{'BBF (t=' + str(config.BBF_T_VALUE) + ')':<15} | {results['bbf']['avg_time']:<20.6f} | {results['bbf']['accuracy']:<12.2f} | {results['bbf']['memory_mb']:.2f}")
            else:
                print(f"No valid results for dimension {d}")
    
    # 汇总所有维度的结果
    if all_dimension_results:
        print("\n" + "="*80)
        print(f"SUMMARY ACROSS ALL DIMENSIONS TESTED: {list(all_dimension_results.keys())}")
        print("="*80)
        
        # 创建汇总表格
        print(f"\n{'Dimension':<10} | {'Algorithm':<15} | {'Avg. Query Time (s)':<20} | {'Accuracy (%)':<12} | {'Memory Usage (MB)':<15}")
        print(f"{'-'*10:<10} | {'-'*15:<15} | {'-'*20:<20} | {'-'*12:<12} | {'-'*15:<15}")
        
        for dim, results in sorted(all_dimension_results.items()):
            bf_time = results['bruteforce']['avg_time']
            kd_time = results['kdtree_exact']['avg_time']
            bbf_time = results['bbf']['avg_time']
            
            # 计算速度比
            bf_speedup = bf_time / kd_time if kd_time > 0 else float('inf')
            kd_speedup = kd_time / bbf_time if bbf_time > 0 else float('inf')
            
            print(f"{dim:<10} | {'Bruteforce':<15} | {bf_time:<20.6f} | {'100.00':<12} | {results['bruteforce']['memory_mb']:.2f}")
            print(f"{'':<10} | {'Standard k-d':<15} | {kd_time:<20.6f} | {results['kdtree_exact']['accuracy']:<12.2f} | {results['kdtree_exact']['memory_mb']:.2f}")
            print(f"{'':<10} | {'BBF (t=' + str(config.BBF_T_VALUE) + ')':<15} | {bbf_time:<20.6f} | {results['bbf']['accuracy']:<12.2f} | {results['bbf']['memory_mb']:.2f}")
            
        # 保存结果到CSV
        if config.SAVE_RESULTS_TO_CSV:
            try:
                save_results_to_csv(all_dimension_results, config.CSV_RESULT_PATH)
                print(f"\nResults saved to {config.CSV_RESULT_PATH}")
            except Exception as e:
                print(f"Error saving results to CSV: {e}")
    
    print("\nPart 1 Experiments Finished.")

if __name__ == '__main__':
    main() 