#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import cv2
import pickle
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Qt错误
import matplotlib.pyplot as plt

# 添加当前目录到路径
sys.path.append('.')

from core.feature_extractor import FeatureExtractor
from core.visual_dictionary import VisualDictionary
from core.improved_bow import ImprovedBoW

class ImprovedRetrievalSystem:
    """
    改进的图像检索系统，专注于解决自匹配问题并提高检索质量
    """
    
    def __init__(self, vocab_size=400, linkage='ward', affinity='euclidean'):
        """
        初始化图像检索系统
        
        参数:
            vocab_size: 视觉词典大小
            linkage: 层次聚类连接方式
            affinity: 距离度量
        """
        self.vocab_size = vocab_size
        self.linkage = linkage
        self.affinity = affinity
        
        # 初始化特征提取器
        self.feature_extractor = None
        
        # 中间文件路径
        self.features_path = f"features/improved_sift_features.pkl"
        self.dictionary_path = f"features/improved_visual_dictionary_sift_{vocab_size}.pkl"
        self.bow_model_path = f"features/improved_bow_model_sift_{vocab_size}.pkl"
        
        # 数据结构
        self.features = {}  # {image_path: (keypoints, descriptors)}
        self.visual_dictionary = None
        self.bow_model = None
        
        # 状态标志
        self._features_loaded = False
        self._dictionary_built = False
        self._bow_computed = False
    
    def setup(self, data_dir="photo", max_features=1000, force_recompute=False):
        """
        设置图像检索系统：提取特征，构建词典，计算词袋表示
        
        参数:
            data_dir: 图像目录
            max_features: 每张图像最多提取的特征点数量
            force_recompute: 是否强制重新计算
            
        返回:
            bool: 成功返回True，否则返回False
        """
        print("设置改进的图像检索系统...")
        
        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor(max_features=max_features)
        
        # 提取图像特征
        if not self._extract_features_from_directory(data_dir, force_recompute):
            print("错误: 提取或加载特征失败")
            return False
        
        # 构建视觉词典
        if not self._build_dictionary(force_recompute):
            print("错误: 构建或加载视觉词典失败")
            return False
        
        # 计算词袋表示
        if not self._compute_bow_representations(force_recompute):
            print("错误: 计算或加载词袋表示失败")
            return False
        
        print("图像检索系统设置成功")
        return True
    
    def is_ready(self):
        """
        检查系统是否已完全设置好，可以进行查询
        
        返回:
            bool: 如果系统已就绪，返回True，否则返回False
        """
        return self._features_loaded and self._dictionary_built and self._bow_computed
    
    def _extract_features_from_directory(self, directory, force_recompute=False):
        """
        从目录中提取SIFT特征
        
        参数:
            directory: 包含图像的目录
            force_recompute: 是否强制重新计算
            
        返回:
            bool: 成功返回True，否则返回False
        """
        try:
            print(f"从{directory}中提取SIFT特征...")
            start_time = time.time()
            
            # 提取特征
            self.features = self.feature_extractor.batch_extract_from_directory(
                directory=directory,
                save_path=self.features_path,
                force_recompute=force_recompute
            )
            
            if not self.features:
                print("警告: 没有提取或加载到特征")
                self._features_loaded = False
                return False
            
            self._features_loaded = True
            print(f"特征提取/加载完成，用时 {time.time() - start_time:.2f} 秒")
            return True
            
        except Exception as e:
            print(f"提取特征出错: {e}")
            self._features_loaded = False
            return False
    
    def _build_dictionary(self, force_recompute=False):
        """
        使用层次聚类构建视觉词典
        
        参数:
            force_recompute: 是否强制重新计算
            
        返回:
            bool: 成功返回True，否则返回False
        """
        try:
            # 如果特征未加载，则跳过
            if not self._features_loaded:
                print("错误: 必须先加载特征才能构建词典")
                return False
            
            # 检查词典是否已存在
            if os.path.exists(self.dictionary_path) and not force_recompute:
                print(f"从{self.dictionary_path}加载预计算的视觉词典")
                try:
                    self.visual_dictionary = VisualDictionary.load(self.dictionary_path)
                    self._dictionary_built = True
                    return True
                except Exception as e:
                    print(f"加载词典出错: {e}")
                    print("将重新计算词典")
            
            print(f"构建包含{self.vocab_size}个词的视觉词典...")
            start_time = time.time()
            
            # 获取所有特征描述符
            all_descriptors = self.feature_extractor.get_all_descriptors(self.features)
            
            if all_descriptors.shape[0] == 0:
                print("警告: 没有找到有效的描述符来构建词典。词典构建跳过。")
                self._dictionary_built = False
                return False
            
            # 构建词典
            self.visual_dictionary = VisualDictionary(
                vocab_size=self.vocab_size,
                linkage=self.linkage,
                affinity=self.affinity
            )
            
            # 拟合词典
            self.visual_dictionary.fit(all_descriptors, verbose=True)
            
            # 保存词典
            self.visual_dictionary.save(self.dictionary_path)
            
            self._dictionary_built = True
            print(f"视觉词典构建完成，用时 {time.time() - start_time:.2f} 秒")
            return True
            
        except Exception as e:
            print(f"构建视觉词典出错: {e}")
            self._dictionary_built = False
            return False
    
    def _compute_bow_representations(self, force_recompute=False):
        """
        计算所有图像的词袋表示
        
        参数:
            force_recompute: 是否强制重新计算
            
        返回:
            bool: 成功返回True，否则返回False
        """
        try:
            # 如果词典未构建，则跳过
            if not self._dictionary_built:
                print("错误: 必须先构建视觉词典才能计算词袋表示")
                return False
            
            # 检查词袋模型是否已存在
            if os.path.exists(self.bow_model_path) and not force_recompute:
                print(f"从{self.bow_model_path}加载预计算的词袋模型")
                try:
                    self.bow_model = ImprovedBoW.load(self.bow_model_path)
                    self._bow_computed = True
                    return True
                except Exception as e:
                    print(f"加载词袋模型出错: {e}")
                    print("将重新计算词袋模型")
            
            print("计算词袋表示...")
            start_time = time.time()
            
            # 创建词袋模型
            self.bow_model = ImprovedBoW(self.visual_dictionary, metric=self.affinity)
            
            # 计算所有图像的词袋表示
            self.bow_model.compute_image_bows(self.features, save_path=self.bow_model_path)
            
            self._bow_computed = True
            print(f"词袋表示计算完成，用时 {time.time() - start_time:.2f} 秒")
            return True
            
        except Exception as e:
            print(f"计算词袋表示出错: {e}")
            self._bow_computed = False
            return False
    
    def query_image_path(self, image_path, top_k=10, sim_method='rerank'):
        """
        使用图像路径查询系统，寻找相似图像
        
        参数:
            image_path: 查询图像的路径
            top_k: 返回的结果数量
            sim_method: 相似度计算方法 ('cosine', 'euclidean', 'combined', 'rerank')
            
        返回:
            list: 最相似图像的列表，格式为 [(image_path, similarity_score), ...]
        """
        # 检查系统是否已设置完成
        if not self.is_ready():
            raise ValueError("错误: 系统未完全设置或词袋模型不可用。请先运行setup()。")
        
        # 从图像中提取特征
        query_image = cv2.imread(image_path)
        
        if query_image is None:
            raise ValueError(f"错误: 无法从{image_path}读取图像")
            
        # 提取SIFT特征
        keypoints, descriptors = self.feature_extractor.extract_features(query_image)
        
        if keypoints is None or descriptors is None or descriptors.shape[0] == 0:
            raise ValueError(f"错误: 无法从{image_path}提取特征")
            
        print(f"\n查询图像: {image_path}")
        print(f"特征点数量: {len(keypoints)}")
        
        # 计算查询图像的词袋表示
        query_bow = self.bow_model.query_image_bow(
            descriptors, 
            keypoints=keypoints, 
            image_shape=query_image.shape
        )
        
        # 处理关键点格式以便于重排序
        pickleable_keypoints = [{
            'pt': (kp.pt[0], kp.pt[1]),
            'size': kp.size,
            'angle': kp.angle,
            'response': kp.response,
            'octave': kp.octave,
            'class_id': kp.class_id
        } for kp in keypoints]
        
        # 搜索相似图像
        print(f"使用相似度方法: {sim_method}")
        results = self.bow_model.search(
            query_path=image_path,
            query_bow=query_bow, 
            query_features=(pickleable_keypoints, descriptors),
            top_k=top_k, 
            sim_method=sim_method
        )
        
        # 检查结果的多样性
        if len(results) > 1:
            scores = [score for _, score in results]
            score_range = max(scores) - min(scores)
            print(f"相似度分数范围: {min(scores):.4f} - {max(scores):.4f} (范围: {score_range:.4f})")
            
            # 检查查询图像自身的排名
            query_rank = -1
            for i, (path, _) in enumerate(results):
                if path == image_path:
                    query_rank = i
                    break
            
            if query_rank >= 0:
                print(f"查询图像自身排名: {query_rank+1} (共{len(results)})")
                if query_rank > 0:
                    print("注意: 查询图像不在第一位，这可能表明检索系统需要进一步优化")
            else:
                print("查询图像不在结果中")
                
        return results
    
    def display_query_results(self, query_path, results, max_display=5, save_path=None):
        """
        Visualize query results
        
        Parameters:
            query_path: Path to the query image
            results: Query results list [(image_path, similarity_score), ...]
            max_display: Maximum number of results to display
            save_path: Path to save the result image
        """
        if len(results) == 0:
            print("No query results to display")
            return
        
        # Read query image
        query_img = cv2.imread(query_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # Limit displayed results
        display_results = results[:min(max_display, len(results))]
        
        # Create subplots
        n_results = len(display_results)
        fig, axes = plt.subplots(1, n_results + 1, figsize=(3 * (n_results + 1), 4))
        
        # Display query image
        axes[0].imshow(query_img)
        axes[0].set_title("Query Image")
        axes[0].axis('off')
        
        # Highlight query image itself
        query_basename = os.path.basename(query_path)
        
        # Display result images
        for i, (img_path, score) in enumerate(display_results):
            result_img = cv2.imread(img_path)
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            
            # Display image
            axes[i+1].imshow(result_img)
            
            # Add border to title if it's the query image itself
            is_self = (img_path == query_path)
            title_text = f"Match {i+1}"
            if is_self:
                title_text += " (Query Image)"
                # Add red border
                axes[i+1].spines['top'].set_color('red')
                axes[i+1].spines['bottom'].set_color('red')
                axes[i+1].spines['left'].set_color('red')
                axes[i+1].spines['right'].set_color('red')
                axes[i+1].spines['top'].set_linewidth(3)
                axes[i+1].spines['bottom'].set_linewidth(3)
                axes[i+1].spines['left'].set_linewidth(3)
                axes[i+1].spines['right'].set_linewidth(3)
            
            axes[i+1].set_title(f"{title_text}\nSimilarity: {score:.4f}")
            axes[i+1].axis('off')
        
        plt.suptitle(f"Query: {os.path.basename(query_path)}")
        plt.tight_layout()
        
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Results saved to {save_path}")
        else:
            # Save to default location
            default_path = f"improved_results/query_{os.path.basename(query_path)}.png"
            os.makedirs(os.path.dirname(default_path), exist_ok=True)
            plt.savefig(default_path)
            print(f"Results saved to {default_path}")
        
        plt.close()

# 使用示例
if __name__ == "__main__":
    import argparse
    
    # 命令行参数
    parser = argparse.ArgumentParser(description="改进的图像检索系统")
    parser.add_argument("--data_dir", type=str, default="photo", help="图像目录")
    parser.add_argument("--vocab_size", type=int, default=400, help="词典大小")
    parser.add_argument("--query", type=str, help="要查询的图像路径")
    parser.add_argument("--force_recompute", action="store_true", help="强制重新计算特征和词典")
    parser.add_argument("--sim_method", type=str, default="rerank", 
                        choices=["cosine", "euclidean", "combined", "rerank"],
                        help="相似度计算方法")
    args = parser.parse_args()
    
    # 创建并设置系统
    system = ImprovedRetrievalSystem(vocab_size=args.vocab_size)
    system.setup(data_dir=args.data_dir, force_recompute=args.force_recompute)
    
    # 如果提供了查询图像，执行查询
    if args.query:
        if os.path.exists(args.query):
            results = system.query_image_path(args.query, sim_method=args.sim_method)
            system.display_query_results(args.query, results)
        else:
            print(f"错误: 查询图像 {args.query} 不存在")
    else:
        print("未提供查询图像。请使用 --query 参数指定查询图像。") 