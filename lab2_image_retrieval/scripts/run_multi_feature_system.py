#!/usr/bin/env python3
import os
import sys
import time
import argparse
import random
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Qt错误
import matplotlib.pyplot as plt

# 添加当前目录到路径
sys.path.append('.')

from systems.multi_feature_retrieval_system import MultiFeatureRetrievalSystem

def main():
    """测试多特征融合的图像检索系统，保证查询图像自身排在第一位"""
    # 命令行参数
    parser = argparse.ArgumentParser(description="测试多特征融合的图像检索系统")
    parser.add_argument("--data_dir", type=str, default="photo", help="图像目录")
    parser.add_argument("--vocab_size", type=int, default=400, help="SIFT词典大小")
    parser.add_argument("--color_bins", type=int, default=16, help="颜色直方图每通道柱数")
    parser.add_argument("--color_space", type=str, default="hsv", choices=["rgb", "hsv"], 
                        help="颜色空间")
    parser.add_argument("--max_features", type=int, default=1000, help="每张图像最多提取的特征点数量")
    parser.add_argument("--force_recompute", action="store_true", help="强制重新计算特征和词典")
    parser.add_argument("--num_queries", type=int, default=5, help="随机查询的样本数量")
    parser.add_argument("--sim_method", type=str, default="rerank", 
                      choices=["cosine", "euclidean", "combined", "rerank"],
                      help="相似度计算方法")
    parser.add_argument("--query_image", type=str, help="指定查询图像的路径")
    parser.add_argument("--sift_weight", type=float, default=0.7, help="SIFT特征权重")
    parser.add_argument("--color_weight", type=float, default=0.3, help="颜色特征权重")
    args = parser.parse_args()
    
    # 创建结果目录
    os.makedirs("multi_feature_results", exist_ok=True)
    
    print("=== 测试多特征融合图像检索系统 (确保自匹配排在第一位) ===")
    print(f"SIFT特征权重: {args.sift_weight}, 颜色特征权重: {args.color_weight}")
    
    # 创建并设置系统
    print(f"创建系统 (SIFT词典大小: {args.vocab_size}, 颜色柱数: {args.color_bins}, 颜色空间: {args.color_space})")
    system = MultiFeatureRetrievalSystem(
        vocab_size=args.vocab_size,
        color_bins=args.color_bins,
        color_space=args.color_space
    )
    
    # 设置特征权重
    system.sift_weight = args.sift_weight
    system.color_weight = args.color_weight
    
    # 设置系统
    setup_start = time.time()
    success = system.setup(
        data_dir=args.data_dir,
        max_features=args.max_features,
        force_recompute=args.force_recompute
    )
    setup_time = time.time() - setup_start
    
    if not success:
        print("系统设置失败")
        return
    
    print(f"系统设置完成，用时 {setup_time:.2f} 秒")
    
    # 如果指定了特定的查询图像
    if args.query_image:
        if os.path.exists(args.query_image):
            print(f"\n执行指定查询: {args.query_image}")
            results = system.query_image_path(args.query_image, sim_method=args.sim_method)
            
            # 显示结果
            save_path = f"multi_feature_results/specific_query_{os.path.basename(args.query_image)}.png"
            system.display_query_results(args.query_image, results, save_path=save_path)
            
            # 分析查询图像的排名
            query_rank = -1
            for i, (path, _) in enumerate(results):
                if path == args.query_image:
                    query_rank = i
                    break
            
            if query_rank == 0:
                print(f"✓ 成功：查询图像 {os.path.basename(args.query_image)} 排在第一位")
            elif query_rank > 0:
                print(f"✗ 失败：查询图像 {os.path.basename(args.query_image)} 排在第 {query_rank+1} 位")
            else:
                print(f"✗ 失败：查询图像 {os.path.basename(args.query_image)} 不在结果中")
            
            return
    
    # 选择随机查询图像
    image_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"错误: 在 {args.data_dir} 中没有找到图像")
        return
    
    # 随机选择查询图像
    random.seed(42)  # 确保可重现性
    query_images = random.sample(image_files, min(args.num_queries, len(image_files)))
    
    # 执行查询
    success_count = 0
    for i, query_path in enumerate(query_images):
        print(f"\n===== 随机查询 {i+1}: {os.path.basename(query_path)} =====")
        
        try:
            # 执行查询
            results = system.query_image_path(query_path, sim_method=args.sim_method)
            
            # 显示结果
            save_path = f"multi_feature_results/random_query_{i+1}_{os.path.basename(query_path)}.png"
            system.display_query_results(query_path, results, save_path=save_path)
            
            # 分析查询图像的排名
            query_rank = -1
            for j, (path, _) in enumerate(results):
                if path == query_path:
                    query_rank = j
                    break
            
            if query_rank == 0:
                print(f"✓ 成功：查询图像排在第一位")
                success_count += 1
            elif query_rank > 0:
                print(f"✗ 失败：查询图像排在第 {query_rank+1} 位")
            else:
                print(f"✗ 失败：查询图像不在结果中")
                
        except Exception as e:
            print(f"查询出错: {e}")
    
    # 打印总体结果
    if args.num_queries > 0:
        success_rate = success_count / len(query_images) * 100
        print(f"\n总体结果: {success_count}/{len(query_images)} 个查询图像排在第一位 ({success_rate:.1f}%)")
        
        if success_rate == 100:
            print("✓ 系统优化成功，所有查询图像都排在第一位！")
        else:
            print("! 系统仍需进一步优化，部分查询图像没有排在第一位。")
            
    # 与之前的改进系统比较
    print("\n=== 多特征融合系统的优势 ===")
    print("1. 融合了SIFT特征和颜色特征，能更全面地表示图像内容")
    print("2. 颜色特征提供了全局信息，弥补了SIFT特征主要捕获局部结构的不足")
    print("3. 通过加权融合机制，可以灵活调整不同特征的重要性")
    print(f"4. 使用{args.color_space}颜色空间的{args.color_bins}柱直方图，捕获更丰富的颜色分布信息")

if __name__ == "__main__":
    main() 