#!/usr/bin/env python3
import numpy as np
import pickle
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time

class VisualDictionary:
    """使用层次聚类方法构建视觉词典"""
    
    def __init__(self, vocab_size=1000, linkage='ward', affinity='euclidean'):
        """
        初始化视觉词典构建器
        
        参数:
            vocab_size: 视觉词典大小（聚类数量）
            linkage: 层次聚类连接方法，如'ward'、'complete'、'average'、'single'
            affinity: 距离度量方法，如'euclidean'、'manhattan'、'cosine'
        """
        self.vocab_size = vocab_size
        self.linkage = linkage
        self.affinity = affinity
        self.clusterer = None
        self.visual_words = None
        self.scaler = StandardScaler()  # 用于特征标准化，提高聚类质量
        
    def fit(self, descriptors, sample_size=None, verbose=True):
        """
        使用特征描述符构建视觉词典
        
        参数:
            descriptors: 特征描述符数组，形状(N, d)
            sample_size: 采样用于聚类的描述符数量，None表示使用全部
            verbose: 是否显示进度信息
            
        返回:
            self: 训练好的视觉词典实例
        """
        if descriptors.shape[0] == 0:
            raise ValueError("描述符数组为空，无法构建词典")
            
        n_samples, n_features = descriptors.shape
        
        if verbose:
            print(f"收集到 {n_samples} 个特征描述符，每个维度 {n_features}")
            
        # 如果描述符数量太多，进行随机采样
        if sample_size and n_samples > sample_size:
            if verbose:
                print(f"随机采样 {sample_size} 个描述符用于构建词典")
            indices = np.random.choice(n_samples, sample_size, replace=False)
            descriptors_sampled = descriptors[indices]
        else:
            descriptors_sampled = descriptors
            
        # 标准化描述符
        if verbose:
            print("标准化描述符...")
        descriptors_scaled = self.scaler.fit_transform(descriptors_sampled)
        
        # 使用层次聚类构建词典
        if verbose:
            print(f"使用层次聚类({self.linkage})构建大小为{self.vocab_size}的视觉词典...")
            start_time = time.time()
            
        self.clusterer = AgglomerativeClustering(
            n_clusters=self.vocab_size,
            linkage=self.linkage,
            affinity=self.affinity
        )
        
        cluster_labels = self.clusterer.fit_predict(descriptors_scaled)
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"聚类完成，用时 {elapsed:.2f} 秒")
            
        # 为每个聚类计算中心（视觉单词）
        if verbose:
            print("计算视觉单词（聚类中心）...")
            
        self.visual_words = np.zeros((self.vocab_size, n_features))
        
        for i in range(self.vocab_size):
            # 找到属于当前聚类的所有描述符
            cluster_descriptors = descriptors_sampled[cluster_labels == i]
            
            if len(cluster_descriptors) > 0:
                # 计算平均值作为聚类中心
                self.visual_words[i] = np.mean(cluster_descriptors, axis=0)
            else:
                # 如果聚类为空（这种情况不应该发生），使用随机描述符
                self.visual_words[i] = descriptors_sampled[np.random.randint(0, len(descriptors_sampled))]
                
        return self
        
    def save(self, filepath):
        """
        保存视觉词典到文件
        
        参数:
            filepath: 保存路径
        """
        if self.visual_words is None:
            raise ValueError("视觉词典尚未构建，无法保存")
            
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'linkage': self.linkage,
                'affinity': self.affinity,
                'visual_words': self.visual_words,
                'scaler': self.scaler
            }, f)
            
        print(f"视觉词典已保存到 {filepath}")
        
    @classmethod
    def load(cls, filepath):
        """
        从文件加载视觉词典
        
        参数:
            filepath: 词典文件路径
            
        返回:
            VisualDictionary: 加载的视觉词典实例
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        vd = cls(
            vocab_size=data['vocab_size'],
            linkage=data['linkage'],
            affinity=data['affinity']
        )
        
        vd.visual_words = data['visual_words']
        vd.scaler = data['scaler']
        
        return vd
        
    def get_visual_words(self):
        """
        获取视觉单词（聚类中心）
        
        返回:
            visual_words: 视觉单词数组，形状(vocab_size, d)
        """
        if self.visual_words is None:
            raise ValueError("视觉词典尚未构建")
            
        return self.visual_words

# 示例用法
if __name__ == "__main__":
    import feature_extractor
    
    # 创建特征提取器并提取特征
    extractor = feature_extractor.FeatureExtractor(method='sift')
    
    # 尝试加载已有特征
    features_path = "features/sift_features.pkl"
    if os.path.exists(features_path):
        print(f"加载已有特征 {features_path}")
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
    else:
        # 从图像中提取特征
        print("从图像中提取特征...")
        features = extractor.extract_from_directory(
            directory="photo",
            save_path=features_path
        )
    
    # 获取所有描述符
    all_descriptors = extractor.get_all_descriptors(features)
    
    # 构建视觉词典
    vocab_size = 200  # 词典大小
    vd = VisualDictionary(vocab_size=vocab_size)
    vd.fit(all_descriptors, sample_size=20000)  # 使用最多20000个描述符进行聚类
    
    # 保存词典
    vd.save("features/visual_dictionary_sift.pkl")
    
    # 显示词典信息
    visual_words = vd.get_visual_words()
    print(f"视觉词典大小: {vocab_size}")
    print(f"视觉单词形状: {visual_words.shape}") 