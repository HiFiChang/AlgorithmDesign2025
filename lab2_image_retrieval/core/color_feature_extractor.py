#!/usr/bin/env python3
import cv2
import numpy as np
import os
import pickle
from tqdm import tqdm

class ColorFeatureExtractor:
    """
    颜色特征提取器，用于提取图像的颜色直方图特征
    """
    
    def __init__(self, bins=16, color_space='hsv'):
        """
        初始化颜色特征提取器
        
        参数:
            bins: 每个颜色通道的直方图柱数
            color_space: 颜色空间，支持'rgb'、'hsv'
        """
        self.bins = bins
        self.color_space = color_space
        self.feature_dim = bins * 3  # 三个颜色通道
    
    def extract_features(self, image):
        """
        从图像中提取颜色直方图特征
        
        参数:
            image: 输入图像 (numpy数组或图像路径)
            
        返回:
            特征向量: 颜色直方图 (一维numpy数组)
        """
        if isinstance(image, str):
            # 加载图像如果给定的是路径
            image = cv2.imread(image)
            if image is None:
                return None
        
        # 转换颜色空间
        if self.color_space.lower() == 'hsv':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.color_space.lower() == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 计算每个通道的直方图
        hist_features = []
        for i in range(3):  # 对每个颜色通道
            hist = cv2.calcHist([image], [i], None, [self.bins], [0, 256])
            # 归一化直方图
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.append(hist)
        
        # 合并三个通道的直方图
        color_features = np.concatenate(hist_features)
        
        return color_features
    
    def batch_extract_from_directory(self, directory, save_path=None, force_recompute=False):
        """
        从目录中提取所有图像的颜色特征
        
        参数:
            directory: 包含图像的目录
            save_path: 保存特征的路径
            force_recompute: 是否强制重新计算
            
        返回:
            features_dict: 特征字典，{image_path: feature_vector}
        """
        # 检查特征是否已经计算并保存
        if save_path and os.path.exists(save_path) and not force_recompute:
            print(f"加载预计算的颜色特征: {save_path}")
            try:
                with open(save_path, 'rb') as f:
                    features = pickle.load(f)
                return features
            except Exception as e:
                print(f"加载特征出错: {e}")
                print("重新计算特征...")
        
        features = {}
        image_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"没有在{directory}中找到图像文件")
            return features
        
        print(f"提取{len(image_files)}张图像的颜色特征...")
        for image_path in tqdm(image_files, desc="处理图像"):
            color_features = self.extract_features(image_path)
            if color_features is not None:
                features[image_path] = color_features
        
        # 保存特征
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(features, f)
            print(f"颜色特征已保存至 {save_path}")
        
        return features
    
# 示例用法
if __name__ == "__main__":
    extractor = ColorFeatureExtractor(bins=16, color_space='hsv')
    features = extractor.batch_extract_from_directory(
        "photo", 
        save_path="features/color_features.pkl"
    )
    print(f"已提取 {len(features)} 张图像的颜色特征")
    
    # 检查特征维度
    for path, feature in list(features.items())[:1]:
        print(f"图像: {path}")
        print(f"特征维度: {feature.shape}")
        print(f"特征前10个元素: {feature[:10]}") 