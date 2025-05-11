#!/usr/bin/env python3
'''
图像检索系统主运行脚本

这个脚本提供了统一的命令行接口来运行不同的图像检索系统:
1. 改进的基于SIFT特征的图像检索系统
2. 多特征融合的图像检索系统（结合SIFT和颜色特征）

使用示例:
    # 运行改进的SIFT特征检索系统
    python image_retrieval.py --system improved --query photo/1.png
    
    # 运行多特征融合检索系统
    python image_retrieval.py --system multi --query photo/1.png --sift_weight 0.6 --color_weight 0.4
'''
import os
import sys
import argparse
import time
import random

# 添加当前目录到路径，确保能够导入子目录中的模块
sys.path.append('.')

# 导入系统
from systems.improved_retrieval_system import ImprovedRetrievalSystem
from systems.multi_feature_retrieval_system import MultiFeatureRetrievalSystem

def main():
    '''主函数'''
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="统一的图像检索系统运行脚本")
    
    # 系统选择
    parser.add_argument("--system", type=str, choices=["improved", "multi"], default="multi",
                      help="要运行的系统类型：improved=改进的SIFT系统，multi=多特征融合系统")
    
    # 数据和查询参数
    parser.add_argument("--data_dir", type=str, default="photo", help="图像目录")
    parser.add_argument("--query", type=str, help="要查询的图像路径")
    parser.add_argument("--random_queries", type=int, default=0, 
                      help="随机执行的查询次数，如果设置，会忽略--query参数")
    
    # 通用系统参数
    parser.add_argument("--force_recompute", action="store_true", help="强制重新计算特征和词典")
    parser.add_argument("--vocab_size", type=int, default=400, help="SIFT词典大小")
    parser.add_argument("--max_features", type=int, default=1000, help="每张图像最多提取的特征点数量")
    parser.add_argument("--sim_method", type=str, default="rerank", 
                      choices=["cosine", "euclidean", "combined", "rerank"],
                      help="相似度计算方法")
    
    # 多特征系统的特定参数
    parser.add_argument("--color_bins", type=int, default=16, help="颜色直方图每通道柱数")
    parser.add_argument("--color_space", type=str, default="hsv", choices=["rgb", "hsv"], 
                      help="颜色空间")
    parser.add_argument("--sift_weight", type=float, default=0.7, help="SIFT特征权重")
    parser.add_argument("--color_weight", type=float, default=0.3, help="颜色特征权重")
    
    args = parser.parse_args()
    
    # 创建结果目录
    if args.system == "improved":
        results_dir = "improved_results"
    else:
        results_dir = "multi_feature_results"
    
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"=== 运行{args.system}图像检索系统 ===")
    
    # 创建并设置系统
    system = None
    if args.system == "improved":
        print(f"创建改进的SIFT系统 (词典大小: {args.vocab_size})")
        system = ImprovedRetrievalSystem(vocab_size=args.vocab_size)
    else:
        print(f"创建多特征融合系统 (SIFT词典大小: {args.vocab_size}, "
              f"颜色柱数: {args.color_bins}, 颜色空间: {args.color_space})")
        system = MultiFeatureRetrievalSystem(
            vocab_size=args.vocab_size,
            color_bins=args.color_bins,
            color_space=args.color_space
        )
        # 设置特征权重
        system.sift_weight = args.sift_weight
        system.color_weight = args.color_weight
        print(f"SIFT特征权重: {args.sift_weight}, 颜色特征权重: {args.color_weight}")
    
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
    
    # 如果指定了随机查询次数
    if args.random_queries > 0:
        # 选择随机查询图像
        image_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"错误: 在 {args.data_dir} 中没有找到图像")
            return
        
        # 随机选择查询图像
        random.seed(42)  # 确保可重现性
        query_images = random.sample(image_files, min(args.random_queries, len(image_files)))
        
        # 执行查询
        success_count = 0
        for i, query_path in enumerate(query_images):
            print(f"\n===== 随机查询 {i+1}: {os.path.basename(query_path)} =====")
            
            try:
                # 执行查询
                results = system.query_image_path(query_path, sim_method=args.sim_method)
                
                # 显示结果
                save_path = f"{results_dir}/random_query_{i+1}_{os.path.basename(query_path)}.png"
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
        success_rate = success_count / len(query_images) * 100
        print(f"\n总体结果: {success_count}/{len(query_images)} 个查询图像排在第一位 ({success_rate:.1f}%)")
        
        if success_rate == 100:
            print("✓ 系统优化成功，所有查询图像都排在第一位！")
        else:
            print("! 系统仍需进一步优化，部分查询图像没有排在第一位。")
    
    # 如果指定了特定的查询图像
    elif args.query:
        if os.path.exists(args.query):
            print(f"\n执行指定查询: {args.query}")
            results = system.query_image_path(args.query, sim_method=args.sim_method)
            
            # 显示结果
            save_path = f"{results_dir}/specific_query_{os.path.basename(args.query)}.png"
            system.display_query_results(args.query, results, save_path=save_path)
            
            # 分析查询图像的排名
            query_rank = -1
            for i, (path, _) in enumerate(results):
                if path == args.query:
                    query_rank = i
                    break
            
            if query_rank == 0:
                print(f"✓ 成功：查询图像 {os.path.basename(args.query)} 排在第一位")
            elif query_rank > 0:
                print(f"✗ 失败：查询图像 {os.path.basename(args.query)} 排在第 {query_rank+1} 位")
            else:
                print(f"✗ 失败：查询图像 {os.path.basename(args.query)} 不在结果中")
        else:
            print(f"错误: 查询图像 {args.query} 不存在")
    else:
        print("未提供查询图像。请使用 --query 参数指定查询图像或 --random_queries 执行随机查询。")

if __name__ == "__main__":
    main()
