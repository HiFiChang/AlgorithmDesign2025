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
from core.color_feature_extractor import ColorFeatureExtractor
from core.visual_dictionary import VisualDictionary
from core.improved_bow import ImprovedBoW

class MultiFeatureRetrievalSystem:
    """
    多特征融合的图像检索系统，结合SIFT特征和颜色特征
    """
    
    def __init__(self, vocab_size=400, color_bins=16, color_space='hsv', 
                linkage='ward', affinity='euclidean'):
        """
        初始化多特征图像检索系统
        
        参数:
            vocab_size: SIFT特征视觉词典大小
            color_bins: 颜色直方图的每通道柱数
            color_space: 颜色空间，支持'rgb'、'hsv'
            linkage: 层次聚类连接方式
            affinity: 距离度量
        """
        self.vocab_size = vocab_size
        self.color_bins = color_bins
        self.color_space = color_space
        self.linkage = linkage
        self.affinity = affinity
        
        # 权重参数
        self.sift_weight = 0.7  # SIFT特征权重
        self.color_weight = 0.3  # 颜色特征权重
        
        # 初始化特征提取器
        self.sift_extractor = None
        self.color_extractor = None
        
        # 中间文件路径
        self.sift_features_path = f"features/improved_sift_features.pkl"
        self.color_features_path = f"features/color_features_{color_bins}_{color_space}.pkl"
        self.dictionary_path = f"features/improved_visual_dictionary_sift_{vocab_size}.pkl"
        self.bow_model_path = f"features/improved_bow_model_sift_{vocab_size}.pkl"
        
        # 数据结构
        self.sift_features = {}  # {image_path: (keypoints, descriptors)}
        self.color_features = {}  # {image_path: color_histogram}
        self.visual_dictionary = None
        self.bow_model = None
        
        # 状态标志
        self._sift_features_loaded = False
        self._color_features_loaded = False
        self._dictionary_built = False
        self._bow_computed = False
    
    def setup(self, data_dir="photo", max_features=1000, force_recompute=False):
        """
        设置图像检索系统：提取特征，构建词典，计算词袋表示
        
        参数:
            data_dir: 图像目录
            max_features: 每张图像最多提取的SIFT特征点数量
            force_recompute: 是否强制重新计算
            
        返回:
            bool: 成功返回True，否则返回False
        """
        print("设置多特征图像检索系统...")
        
        # 初始化特征提取器
        self.sift_extractor = FeatureExtractor(max_features=max_features)
        self.color_extractor = ColorFeatureExtractor(bins=self.color_bins, color_space=self.color_space)
        
        # 提取SIFT特征
        if not self._extract_sift_features(data_dir, force_recompute):
            print("错误: 提取或加载SIFT特征失败")
            return False
        
        # 提取颜色特征
        if not self._extract_color_features(data_dir, force_recompute):
            print("错误: 提取或加载颜色特征失败")
            return False
        
        # 构建视觉词典
        if not self._build_dictionary(force_recompute):
            print("错误: 构建或加载视觉词典失败")
            return False
        
        # 计算词袋表示
        if not self._compute_bow_representations(force_recompute):
            print("错误: 计算或加载词袋表示失败")
            return False
        
        print("多特征图像检索系统设置成功")
        return True
    
    def is_ready(self):
        """
        检查系统是否已完全设置好，可以进行查询
        
        返回:
            bool: 如果系统已就绪，返回True，否则返回False
        """
        return (self._sift_features_loaded and self._color_features_loaded and 
                self._dictionary_built and self._bow_computed)
    
    def _extract_sift_features(self, directory, force_recompute=False):
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
            self.sift_features = self.sift_extractor.batch_extract_from_directory(
                directory=directory,
                save_path=self.sift_features_path,
                force_recompute=force_recompute
            )
            
            if not self.sift_features:
                print("警告: 没有提取或加载到SIFT特征")
                self._sift_features_loaded = False
                return False
            
            self._sift_features_loaded = True
            print(f"SIFT特征提取/加载完成，用时 {time.time() - start_time:.2f} 秒")
            return True
            
        except Exception as e:
            print(f"提取SIFT特征出错: {e}")
            self._sift_features_loaded = False
            return False
    
    def _extract_color_features(self, directory, force_recompute=False):
        """
        从目录中提取颜色特征
        
        参数:
            directory: 包含图像的目录
            force_recompute: 是否强制重新计算
            
        返回:
            bool: 成功返回True，否则返回False
        """
        try:
            print(f"从{directory}中提取颜色特征...")
            start_time = time.time()
            
            # 提取特征
            self.color_features = self.color_extractor.batch_extract_from_directory(
                directory=directory,
                save_path=self.color_features_path,
                force_recompute=force_recompute
            )
            
            if not self.color_features:
                print("警告: 没有提取或加载到颜色特征")
                self._color_features_loaded = False
                return False
            
            self._color_features_loaded = True
            print(f"颜色特征提取/加载完成，用时 {time.time() - start_time:.2f} 秒")
            return True
            
        except Exception as e:
            print(f"提取颜色特征出错: {e}")
            self._color_features_loaded = False
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
            if not self._sift_features_loaded:
                print("错误: 必须先加载SIFT特征才能构建词典")
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
            all_descriptors = self.sift_extractor.get_all_descriptors(self.sift_features)
            
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
            self.bow_model.compute_image_bows(self.sift_features, save_path=self.bow_model_path)
            
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
            raise ValueError("错误: 系统未完全设置。请先运行setup()。")
        
        # 从图像中提取SIFT特征
        query_image = cv2.imread(image_path)
        if query_image is None:
            raise ValueError(f"错误: 无法从{image_path}读取图像")
            
        # 提取SIFT特征
        sift_keypoints, sift_descriptors = self.sift_extractor.extract_features(query_image)
        if sift_descriptors is None or sift_descriptors.shape[0] == 0:
            print(f"警告: 无法从{image_path}提取SIFT特征")
            sift_descriptors = np.zeros((1, 128))  # 创建空特征
        
        # 提取颜色特征
        color_features = self.color_extractor.extract_features(query_image)
        if color_features is None:
            print(f"警告: 无法从{image_path}提取颜色特征")
            color_features = np.zeros(self.color_bins * 3)  # 创建空特征
            
        print(f"\n查询图像: {image_path}")
        print(f"SIFT特征点数量: {len(sift_keypoints)}")
        print(f"颜色特征维度: {len(color_features)}")
        
        # 计算查询图像的词袋表示
        query_bow = self.bow_model.query_image_bow(
            sift_descriptors, 
            keypoints=sift_keypoints, 
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
        } for kp in sift_keypoints]
        
        # 搜索相似图像（SIFT特征）
        print(f"使用相似度方法: {sim_method}")
        sift_results = self.bow_model.search(
            query_path=image_path,
            query_bow=query_bow, 
            query_features=(pickleable_keypoints, sift_descriptors),
            top_k=top_k * 2,  # 获取更多候选以便后续融合
            sim_method=sim_method
        )
        
        # 颜色特征相似度计算
        color_similarities = self._compute_color_similarities(image_path, color_features)
        
        # 融合SIFT和颜色相似度
        fusion_results = self._fuse_similarities(image_path, sift_results, color_similarities, top_k)
        
        return fusion_results
    
    def _compute_color_similarities(self, query_path, query_color):
        """
        计算查询图像与数据库中所有图像的颜色相似度
        
        参数:
            query_path: 查询图像路径
            query_color: 查询图像的颜色特征
            
        返回:
            dict: 图像路径到相似度的映射
        """
        similarities = {}
        
        for path, color_feature in self.color_features.items():
            # 计算余弦相似度
            similarity = 1 - np.sum((query_color - color_feature) ** 2) / (
                np.sqrt(np.sum(query_color ** 2)) * np.sqrt(np.sum(color_feature ** 2))
            )
            
            # 确保自匹配情况下，相似度为1.0
            if path == query_path:
                similarity = 1.0
                
            similarities[path] = similarity
        
        return similarities
    
    def _fuse_similarities(self, query_path, sift_results, color_similarities, top_k):
        """
        融合SIFT和颜色相似度
        
        参数:
            query_path: 查询图像路径
            sift_results: SIFT特征的相似度结果
            color_similarities: 颜色特征的相似度结果
            top_k: 返回的结果数量
            
        返回:
            list: 融合后的相似图像列表
        """
        # 转换SIFT结果为字典
        sift_dict = {path: score for path, score in sift_results}
        
        # 共同图像集合
        common_images = set(sift_dict.keys()).intersection(set(color_similarities.keys()))
        
        # 融合相似度
        fusion_scores = {}
        for path in common_images:
            sift_score = sift_dict.get(path, 0)
            color_score = color_similarities.get(path, 0)
            
            # 加权融合
            fusion_score = self.sift_weight * sift_score + self.color_weight * color_score
            
            # 确保查询图像自身排在第一位
            if path == query_path:
                fusion_score = 1.0
                
            fusion_scores[path] = fusion_score
        
        # 按融合得分排序
        sorted_results = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 确保查询图像在第一位
        if query_path in fusion_scores:
            query_idx = -1
            for i, (path, _) in enumerate(sorted_results):
                if path == query_path:
                    query_idx = i
                    break
            
            if query_idx > 0:
                query_item = sorted_results.pop(query_idx)
                sorted_results.insert(0, query_item)
        
        return sorted_results[:top_k]
    
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
            default_path = f"multi_feature_results/query_{os.path.basename(query_path)}.png"
            os.makedirs(os.path.dirname(default_path), exist_ok=True)
            plt.savefig(default_path)
            print(f"Results saved to {default_path}")
        
        plt.close()

# 使用示例
if __name__ == "__main__":
    import argparse
    
    # 命令行参数
    parser = argparse.ArgumentParser(description="多特征融合的图像检索系统")
    parser.add_argument("--data_dir", type=str, default="photo", help="图像目录")
    parser.add_argument("--vocab_size", type=int, default=400, help="SIFT词典大小")
    parser.add_argument("--color_bins", type=int, default=16, help="颜色直方图每通道柱数")
    parser.add_argument("--color_space", type=str, default="hsv", choices=["rgb", "hsv"], 
                        help="颜色空间")
    parser.add_argument("--query", type=str, help="要查询的图像路径")
    parser.add_argument("--force_recompute", action="store_true", help="强制重新计算特征和词典")
    parser.add_argument("--sim_method", type=str, default="rerank", 
                        choices=["cosine", "euclidean", "combined", "rerank"],
                        help="相似度计算方法")
    parser.add_argument("--sift_weight", type=float, default=0.7, help="SIFT特征权重")
    parser.add_argument("--color_weight", type=float, default=0.3, help="颜色特征权重")
    args = parser.parse_args()
    
    # 创建并设置系统
    system = MultiFeatureRetrievalSystem(
        vocab_size=args.vocab_size,
        color_bins=args.color_bins,
        color_space=args.color_space
    )
    
    # 设置特征权重
    system.sift_weight = args.sift_weight
    system.color_weight = args.color_weight
    
    # 设置系统
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