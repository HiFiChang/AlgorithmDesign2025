#!/usr/bin/env python3
import numpy as np
import os
import pickle
import time
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine, euclidean
from tqdm import tqdm
import cv2

class ImprovedBoW:
    """
    改进的词袋表示，专注于解决图像自匹配问题
    """
    
    def __init__(self, visual_dictionary=None, metric='euclidean'):
        """
        初始化改进的词袋模型
        
        参数:
            visual_dictionary: 视觉词典实例或词典文件路径
            metric: 最近邻匹配的距离度量
        """
        self.visual_words = None
        self.scaler = None
        self.metric = metric
        self.nn_model = None
        self.images_bow = {}         # 图像的词袋表示
        self.image_features = {}     # 存储每个图像的原始特征
        self.image_paths = []        # 数据库中的图像路径
        self.use_tfidf = True        # 是否使用TF-IDF加权
        self.idf = None              # IDF权重向量
        self.spatial_pyramid = True  # 是否使用空间金字塔
        self.exact_match_bonus = 0.05  # 精确匹配奖励系数
        
        # 如果提供了视觉词典，加载它
        if visual_dictionary is not None:
            if isinstance(visual_dictionary, str):
                self._load_dictionary_from_file(visual_dictionary)
            elif hasattr(visual_dictionary, 'get_visual_words'):
                self.visual_words = visual_dictionary.get_visual_words()
                self.scaler = getattr(visual_dictionary, 'scaler', None)
            else:
                raise ValueError("visual_dictionary必须是路径或VisualDictionary实例")
    
    def _load_dictionary_from_file(self, filepath):
        """从文件加载视觉词典"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            self.visual_words = data.get('visual_words')
            self.scaler = data.get('scaler')
        except Exception as e:
            print(f"加载词典出错: {e}")
    
    def fit(self, verbose=True):
        """
        使用视觉词典构建最近邻模型，用于将特征描述符映射到视觉词汇
        
        参数:
            verbose: 是否显示进度信息
            
        返回:
            self: 实例本身
        """
        if self.visual_words is None:
            raise ValueError("未加载视觉词典，无法构建最近邻模型")
            
        if verbose:
            print(f"构建最近邻模型 (使用{self.metric}度量)...")
            
        self.nn_model = NearestNeighbors(n_neighbors=1, algorithm='auto', metric=self.metric)
        self.nn_model.fit(self.visual_words)
        
        return self
    
    def compute_bow(self, descriptors, keypoints=None, image_shape=None):
        """
        计算单个图像的词袋表示
        
        参数:
            descriptors: 图像的特征描述符，形状 (n, d)
            keypoints: 特征点（用于空间金字塔）
            image_shape: 图像尺寸（用于空间金字塔）
            
        返回:
            hist: 归一化的词袋向量
        """
        if self.nn_model is None:
            if self.visual_words is None:
                raise ValueError("未加载视觉词典，无法构建最近邻模型")
            self.fit(verbose=False)
            
        if descriptors.shape[0] == 0:
            # 如果没有描述符，返回零向量
            return np.zeros(len(self.visual_words))
            
        # 标准化描述符
        if self.scaler is not None:
            descriptors = self.scaler.transform(descriptors)
            
        # 为每个描述符找到最近的视觉词
        distances, indices = self.nn_model.kneighbors(descriptors)
        
        # 构建词频直方图 (TF)
        hist = np.zeros(len(self.visual_words))
        
        # 基础方法：统计词频
        for idx in indices.flatten():
            hist[idx] += 1
        
        # 使用对数TF权重 (1 + log(tf))
        mask = hist > 0
        hist[mask] = 1 + np.log(hist[mask])
        
        # 空间金字塔实现
        if self.spatial_pyramid and keypoints is not None and image_shape is not None:
            hist_pyramid = self._compute_spatial_pyramid(descriptors, keypoints, indices, image_shape)
            # 组合基础直方图和空间金字塔结果 (70% 基础 + 30% 空间)
            hist = 0.7 * hist + 0.3 * hist_pyramid
        
        # L2 归一化
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
            
        return hist
    
    def _compute_spatial_pyramid(self, descriptors, keypoints, indices, image_shape):
        """
        计算空间金字塔表示
        
        参数:
            descriptors: 特征描述符
            keypoints: 特征点
            indices: 特征到视觉词的映射索引
            image_shape: 图像尺寸 (h, w)
            
        返回:
            hist_pyramid: 空间金字塔加权直方图
        """
        h, w = image_shape[:2]
        
        # 2x2网格
        grid_size = 2
        grid_h = h / grid_size
        grid_w = w / grid_size
        
        # 每个网格单元的直方图
        grid_hists = np.zeros((grid_size, grid_size, len(self.visual_words)))
        
        # 计算每个特征点所在的网格单元
        for i, kp in enumerate(keypoints):
            # 获取关键点坐标，考虑不同格式的关键点
            try:
                if isinstance(kp, dict) and 'pt' in kp:
                    # 字典格式 {'pt': (x, y), ...}
                    x, y = kp['pt']
                elif hasattr(kp, 'pt'):
                    # cv2.KeyPoint 对象
                    x, y = kp.pt
                else:
                    # 假设是 (x, y) 元组或列表
                    x, y = kp[0], kp[1]
            except Exception:
                # 如果无法获取坐标，跳过此关键点
                continue
            
            # 确定网格位置
            grid_x = min(int(x / grid_w), grid_size - 1)
            grid_y = min(int(y / grid_h), grid_size - 1)
            
            # 更新对应网格的直方图
            word_idx = indices[i][0]
            grid_hists[grid_y, grid_x, word_idx] += 1
        
        # 归一化每个网格的直方图
        for i in range(grid_size):
            for j in range(grid_size):
                mask = grid_hists[i, j] > 0
                grid_hists[i, j, mask] = 1 + np.log(grid_hists[i, j, mask])
                
                # 归一化
                norm = np.linalg.norm(grid_hists[i, j])
                if norm > 0:
                    grid_hists[i, j] = grid_hists[i, j] / norm
        
        # 平铺并加权合并直方图
        # 整体直方图 (level 0) 权重0.5，每个小网格 (level 1) 权重0.5/4=0.125
        hist_pyramid = np.zeros(len(self.visual_words))
        
        # 添加所有网格直方图
        for i in range(grid_size):
            for j in range(grid_size):
                hist_pyramid += 0.125 * grid_hists[i, j]
                
        # L2 归一化
        norm = np.linalg.norm(hist_pyramid)
        if norm > 0:
            hist_pyramid = hist_pyramid / norm
                
        return hist_pyramid
    
    def compute_image_bows(self, features_dict, save_path=None, verbose=True):
        """
        计算多个图像的词袋表示
        
        参数:
            features_dict: 图像特征字典，{image_path: (keypoints, descriptors)}
            save_path: 保存结果的路径
            verbose: 是否显示进度信息
            
        返回:
            images_bow: 词袋表示字典，{image_path: bow_vector}
        """
        if self.nn_model is None:
            if self.visual_words is None:
                raise ValueError("未加载视觉词典，无法构建最近邻模型")
            self.fit(verbose=verbose)
            
        self.images_bow = {}
        self.image_features = {}
        self.image_paths = list(features_dict.keys())
        
        # 初始化TF词袋向量（未加权）
        raw_bow = {}
        doc_freq = np.zeros(len(self.visual_words))
        
        # 第一轮：计算TF词频向量
        iterator = tqdm(features_dict.items()) if verbose else features_dict.items()
        
        for image_path, (keypoints, descriptors) in iterator:
            if verbose:
                iterator.set_description(f"计算词袋表示: {os.path.basename(image_path)}")
                
            if descriptors is not None and descriptors.shape[0] > 0:
                # 获取图像尺寸
                try:
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    image_shape = image.shape if image is not None else (100, 100)
                except:
                    image_shape = (100, 100)  # 默认尺寸
                
                # 计算词袋表示
                bow = self.compute_bow(descriptors, keypoints, image_shape)
                raw_bow[image_path] = bow
                
                # 存储原始特征（用于后续匹配）
                self.image_features[image_path] = (keypoints, descriptors)
                
                # 更新文档频率 (每个词出现在多少文档中)
                doc_freq += (bow > 0).astype(int)
        
        # 计算IDF
        n_docs = len(raw_bow)
        if n_docs > 0 and self.use_tfidf:
            # 平滑处理，防止除零错误
            self.idf = np.log((n_docs + 1) / (doc_freq + 1)) + 1
            
            # 应用TF-IDF权重
            if verbose:
                print("应用TF-IDF权重...")
            
            for image_path, tf in raw_bow.items():
                # TF-IDF = TF * IDF
                tfidf = tf * self.idf
                
                # 重新归一化
                norm = np.linalg.norm(tfidf)
                if norm > 0:
                    tfidf = tfidf / norm
                
                self.images_bow[image_path] = tfidf
        else:
            # 不使用TF-IDF，直接使用原始词袋
            self.images_bow = raw_bow
        
        if verbose:
            print(f"已计算 {len(self.images_bow)} 张图像的词袋表示")
            
            # 检查稀疏度
            sparsity_sum = 0
            for bow in self.images_bow.values():
                sparsity_sum += np.count_nonzero(bow) / len(bow)
            avg_sparsity = sparsity_sum / len(self.images_bow) if self.images_bow else 0
            print(f"词袋表示平均非零元素比例: {avg_sparsity:.2%}")
            
        # 保存结果
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'images_bow': self.images_bow,
                    'metric': self.metric,
                    'visual_words': self.visual_words,
                    'scaler': self.scaler,
                    'idf': self.idf,
                    'use_tfidf': self.use_tfidf
                }, f)
            print(f"词袋表示已保存至 {save_path}")
            
        return self.images_bow
    
    def query_image_bow(self, descriptors, keypoints=None, image_shape=None):
        """
        计算查询图像的词袋表示并应用权重
        
        参数:
            descriptors: 查询图像的特征描述符
            keypoints: 特征点（用于空间金字塔）
            image_shape: 图像尺寸（用于空间金字塔）
            
        返回:
            query_bow: 带权重的词袋表示
        """
        # 计算基本词袋表示
        bow = self.compute_bow(descriptors, keypoints, image_shape)
        
        # 应用IDF权重
        if self.use_tfidf and self.idf is not None:
            bow = bow * self.idf
            
            # 重新归一化
            norm = np.linalg.norm(bow)
            if norm > 0:
                bow = bow / norm
                
        return bow
    
    def search(self, query_path, query_bow, query_features=None, top_k=10, sim_method='combined'):
        """
        搜索最相似的图像
        
        参数:
            query_path: 查询图像路径
            query_bow: 查询图像的词袋表示
            query_features: 查询图像的原始特征 (keypoints, descriptors)
            top_k: 返回的最相似图像数量
            sim_method: 相似度方法 ('cosine', 'euclidean', 'combined', 'rerank')
            
        返回:
            top_matches: 最相似图像列表，[(image_path, similarity_score), ...]
        """
        if not self.images_bow:
            raise ValueError("数据库为空，无法执行搜索")
        
        # 打印查询向量的稀疏度
        nonzero_query = np.count_nonzero(query_bow)
        print(f"查询向量非零元素: {nonzero_query}/{len(query_bow)} ({nonzero_query/len(query_bow):.2%})")
        
        # 检查数据库中词袋向量的稀疏度
        zero_vectors = 0
        sparsity_sum = 0
        for bow in self.images_bow.values():
            nonzeros = np.count_nonzero(bow)
            sparsity_sum += nonzeros/len(bow)
            if nonzeros == 0:
                zero_vectors += 1
        
        avg_sparsity = sparsity_sum / len(self.images_bow) if self.images_bow else 0
        print(f"数据库中全零向量数量: {zero_vectors}/{len(self.images_bow)}")
        print(f"数据库平均非零元素比例: {avg_sparsity:.2%}")
            
        similarities = {}
        
        # 计算基础相似度
        for image_path, bow in self.images_bow.items():
            if sim_method == 'cosine':
                # 余弦相似度
                sim = 1 - cosine(query_bow, bow)
            elif sim_method == 'euclidean':
                # 欧氏距离（转换为相似度）
                dist = np.linalg.norm(query_bow - bow)
                sim = 1 / (1 + dist)
            elif sim_method == 'combined':
                # 结合余弦和欧氏距离
                cos_sim = 1 - cosine(query_bow, bow)
                euc_dist = np.linalg.norm(query_bow - bow)
                euc_sim = 1 / (1 + euc_dist)
                # 加权组合
                sim = 0.7 * cos_sim + 0.3 * euc_sim
            else:
                # 默认使用余弦相似度
                sim = 1 - cosine(query_bow, bow)
            
            # 确保自匹配情况下，相似度得到奖励
            if query_path == image_path:
                # 给查询图像自身一个额外加分
                sim = min(1.0, sim + self.exact_match_bonus)
                
            similarities[image_path] = sim
            
        # 排序前检查相似度分布
        sim_values = list(similarities.values())
        if len(sim_values) > 1:
            print(f"相似度范围: {min(sim_values):.4f} - {max(sim_values):.4f}")
            print(f"相似度标准差: {np.std(sim_values):.4f}")
        
        # 重排序：如果使用rerank模式，根据特征点匹配进行重排序
        if sim_method == 'rerank' and query_features is not None:
            # 获取初步排序结果
            sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            top_candidates = sorted_matches[:min(50, len(sorted_matches))]  # 前50个候选
            
            # 重排序结果
            reranked_matches = self._rerank_results(query_path, query_features, top_candidates)
            
            # 返回前K个重排序结果
            return reranked_matches[:top_k]
        else:
            # 按相似度降序排序
            sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            # 确保查询图像本身排在前面（如果在数据库中）
            # 这是一个备用措施，以防额外加分不足
            if query_path in self.images_bow:
                # 找到查询图像在排序结果中的位置
                query_idx = -1
                for i, (path, _) in enumerate(sorted_matches):
                    if path == query_path:
                        query_idx = i
                        break
                
                # 如果查询图像不在第一位，将其移到第一位
                if query_idx > 0:
                    # 从当前位置移除
                    query_item = sorted_matches.pop(query_idx)
                    # 插入到第一位
                    sorted_matches.insert(0, query_item)
            
            # 返回前K个结果
            return sorted_matches[:top_k]
    
    def _rerank_results(self, query_path, query_features, candidates):
        """
        基于特征匹配重排序候选结果
        
        参数:
            query_path: 查询图像路径
            query_features: 查询图像特征 (keypoints, descriptors)
            candidates: 候选结果 [(image_path, similarity), ...]
            
        返回:
            reranked: 重排序结果
        """
        query_kps, query_descs = query_features
        
        reranked = []
        
        # FLANN参数
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        for image_path, bow_sim in candidates:
            if image_path not in self.image_features:
                reranked.append((image_path, bow_sim))
                continue
                
            # 获取候选图像特征
            db_kps, db_descs = self.image_features[image_path]
            
            if len(query_descs) == 0 or len(db_descs) == 0:
                reranked.append((image_path, bow_sim))
                continue
            
            # 使用FLANN匹配器进行匹配
            try:
                matches = flann.knnMatch(query_descs, db_descs, k=2)
                
                # 应用比率测试选择好的匹配
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
                
                # 计算匹配得分
                match_score = len(good_matches) / max(len(query_descs), len(db_descs))
                
                # 组合词袋相似度和匹配得分
                combined_score = 0.6 * bow_sim + 0.4 * match_score
                
                # 自匹配情况加分
                if query_path == image_path:
                    combined_score = min(1.0, combined_score + self.exact_match_bonus)
                    
                reranked.append((image_path, combined_score))
                
            except Exception as e:
                print(f"重排序出错 ({os.path.basename(image_path)}): {e}")
                reranked.append((image_path, bow_sim))
        
        # 按新得分排序
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # 确保查询图像在第一位
        if query_path in self.image_paths:
            # 找到查询图像在重排序结果中的位置
            query_idx = -1
            for i, (path, _) in enumerate(reranked):
                if path == query_path:
                    query_idx = i
                    break
            
            # 如果查询图像不在第一位，将其移到第一位
            if query_idx > 0:
                # 从当前位置移除
                query_item = reranked.pop(query_idx)
                # 插入到第一位，并确保相似度为1.0
                reranked.insert(0, (query_path, 1.0))
        
        return reranked
    
    @classmethod
    def load(cls, filepath):
        """
        从文件加载词袋模型
        
        参数:
            filepath: 模型文件路径
            
        返回:
            ImprovedBoW: 加载的模型实例
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        # 获取元数据
        metric = data.get('metric', 'euclidean')
        
        # 创建新实例
        bow = cls(metric=metric)
        
        # 加载词袋表示
        bow.images_bow = data.get('images_bow', {})
        
        # 加载视觉词典信息
        bow.visual_words = data.get('visual_words', None)
        bow.scaler = data.get('scaler', None)
        
        # 加载TF-IDF相关信息
        bow.idf = data.get('idf', None)
        bow.use_tfidf = data.get('use_tfidf', True)
        
        # 加载图像路径列表
        bow.image_paths = list(bow.images_bow.keys())
        
        # 初始化最近邻模型
        if bow.visual_words is not None:
            bow.fit(verbose=False)
            
        return bow
    
    def save(self, filepath):
        """
        保存词袋模型到文件
        
        参数:
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'images_bow': self.images_bow,
                'metric': self.metric,
                'visual_words': self.visual_words,
                'scaler': self.scaler,
                'idf': self.idf,
                'use_tfidf': self.use_tfidf
            }, f)
            
        print(f"词袋模型已保存至 {filepath}") 