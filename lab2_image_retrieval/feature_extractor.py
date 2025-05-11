#!/usr/bin/env python3
import cv2
import os
import numpy as np
import pickle
from tqdm import tqdm

class FeatureExtractor:
    """
    A class to extract SIFT features from images.
    """
    
    def __init__(self, max_features=500):
        """
        Initialize the feature extractor with SIFT.
        
        Args:
            max_features: Maximum number of features to extract per image
        """
        self.detector = cv2.SIFT_create(nfeatures=max_features)
        self.max_features = max_features
    
    def extract_features(self, image):
        """
        Extract SIFT features from a given image.
        
        Args:
            image: Input image (numpy array or path to image file)
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        if isinstance(image, str):
            # Load image if a path is provided
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None, None
        elif len(image.shape) == 3:
            # Convert to grayscale if color image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract keypoints and descriptors
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        
        if keypoints is None or len(keypoints) == 0:
            return None, None
            
        return keypoints, descriptors
    
    def extract_from_file(self, image_path):
        """
        Extract features from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                return None, None
                
            return self.extract_features(image)
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None, None
    
    def batch_extract_from_directory(self, directory, save_path=None, force_recompute=False):
        """
        Extract features from all images in a directory.
        
        Args:
            directory: Directory containing image files
            save_path: Path to save the extracted features
            force_recompute: Whether to force recomputation of features
            
        Returns:
            dict: Dictionary mapping image paths to (keypoints, descriptors) tuples
        """
        # Check if features have already been computed and saved
        if save_path and os.path.exists(save_path) and not force_recompute:
            print(f"Loading pre-computed features from {save_path}")
            try:
                with open(save_path, 'rb') as f:
                    features = pickle.load(f)
                return features
            except Exception as e:
                print(f"Error loading features from {save_path}: {e}")
                print("Recomputing features...")
        
        features = {}
        image_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No image files found in {directory}")
            return features
        
        print(f"Extracting features from {len(image_files)} images...")
        valid_feature_count = 0
        
        for image_path in tqdm(image_files, desc="Processing images"):
            keypoints, descriptors = self.extract_from_file(image_path)
            
            if keypoints is not None and descriptors is not None:
                # Convert keypoints to a pickleable format
                keypoints_pickleable = self._keypoints_to_pickleable(keypoints)
                features[image_path] = (keypoints_pickleable, descriptors)
                valid_feature_count += 1
        
        print(f"Processed {len(image_files)} images, {valid_feature_count} had valid feature descriptors")
        
        # Save features if a save path is provided
        if save_path and valid_feature_count > 0:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(features, f)
            print(f"Features saved to {save_path}")
        
        return features
    
    def get_all_descriptors(self, features):
        """
        Get all descriptors from a features dictionary.
        
        Args:
            features: Dictionary mapping image paths to (keypoints, descriptors) tuples
            
        Returns:
            numpy.ndarray: All descriptors concatenated
        """
        if not features:
            return np.array([])
        
        all_descriptors = []
        for kp, desc in features.values():
            if desc is not None and desc.shape[0] > 0:
                all_descriptors.append(desc)
        
        if not all_descriptors:
            return np.array([])
        
        return np.vstack(all_descriptors)
    
    def _keypoints_to_pickleable(self, keypoints):
        """
        Convert cv2.KeyPoint objects to a pickleable format.
        
        Args:
            keypoints: List of cv2.KeyPoint objects
            
        Returns:
            list: List of dictionaries representing keypoints
        """
        return [{
            'pt': (kp.pt[0], kp.pt[1]),
            'size': kp.size,
            'angle': kp.angle,
            'response': kp.response,
            'octave': kp.octave,
            'class_id': kp.class_id
        } for kp in keypoints]
    
    def _pickleable_to_keypoints(self, pickled_keypoints):
        """
        Convert pickleable format back to cv2.KeyPoint objects.
        
        Args:
            pickled_keypoints: List of dictionaries representing keypoints
            
        Returns:
            list: List of cv2.KeyPoint objects
        """
        keypoints = []
        for pkp in pickled_keypoints:
            try:
                # Check that all needed keys are present
                if not all(k in pkp for k in ['pt', 'size', 'angle', 'response', 'octave', 'class_id']):
                    print(f"Warning: KeyPoint dictionary missing required keys: {pkp}")
                    continue
                
                # Using positional arguments to construct KeyPoint
                # Format: cv2.KeyPoint(x, y, size, angle, response, octave, class_id)
                x, y = pkp['pt']
                kp = cv2.KeyPoint(
                    x=float(x),
                    y=float(y), 
                    size=float(pkp['size']),
                    angle=float(pkp['angle']), 
                    response=float(pkp['response']), 
                    octave=int(pkp['octave']), 
                    class_id=int(pkp['class_id'])
                )
                keypoints.append(kp)
            except Exception as e:
                print(f"Error converting pickled keypoint to KeyPoint: {e}, data: {pkp}")
                continue
                
        return keypoints

# 示例用法
if __name__ == "__main__":
    # 创建特征提取器
    extractor = FeatureExtractor(max_features=500)
    
    # 从目录中提取特征
    features = extractor.batch_extract_from_directory(
        directory="photo",
        save_path="features/sift_features.pkl"
    )
    
    # 获取所有描述符
    all_desc = extractor.get_all_descriptors(features)
    print(f"总共提取了 {len(all_desc)} 个特征描述符")
    
    # 可以查看一个图像的特征
    for image_name, (keypoints, descriptors) in list(features.items())[:1]:
        print(f"\n图像 {image_name}:")
        print(f"  - 关键点数量: {len(keypoints)}")
        if len(descriptors) > 0:
            print(f"  - 描述符大小: {descriptors.shape}") 