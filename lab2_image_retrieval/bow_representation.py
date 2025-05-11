#!/usr/bin/env python3
import numpy as np
import pickle
import os
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
from tqdm import tqdm
import time
from visual_dictionary import VisualDictionary

class BagOfWords:
    """Bag of Words model representation based on visual dictionary"""
    
    def __init__(self, visual_dictionary=None, metric='euclidean'):
        """
        Initialize the Bag of Words model
        
        Args:
            visual_dictionary: VisualDictionary instance or path to dictionary file
            metric: Distance metric for nearest neighbor matching, such as 'euclidean', 'cosine', etc.
        """
        self.visual_words = None
        self.scaler = None
        self.metric = metric
        self.nn_model = None
        self.images_bow = {}  # Store BoW representations of images
        
        # If a visual dictionary is provided, load it
        if visual_dictionary is not None:
            if isinstance(visual_dictionary, str):
                self._load_dictionary_from_file(visual_dictionary)
            elif hasattr(visual_dictionary, 'get_visual_words'):
                self.visual_words = visual_dictionary.get_visual_words()
                self.scaler = getattr(visual_dictionary, 'scaler', None)
            else:
                raise ValueError("visual_dictionary must be a path or a VisualDictionary instance")
                
    def _load_dictionary_from_file(self, filepath):
        """Load visual dictionary from file"""
        try:
            # Try to load as a VisualDictionary
            vd = VisualDictionary.load(filepath)
            self.visual_words = vd.get_visual_words()
            self.scaler = vd.scaler
        except Exception as e:
            print(f"Error loading as VisualDictionary: {e}")
            # Try to load as a direct dictionary file
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            self.visual_words = data.get('visual_words')
            self.scaler = data.get('scaler')
            
    def fit(self, verbose=True):
        """
        Build a nearest neighbor model using the visual dictionary to map feature descriptors to visual words
        
        Args:
            verbose: Whether to show progress information
            
        Returns:
            self: Instance with built nearest neighbor model
        """
        if self.visual_words is None:
            raise ValueError("Visual dictionary not loaded, cannot build nearest neighbor model")
            
        if verbose:
            print(f"Building nearest neighbor model using {self.metric} metric...")
            
        self.nn_model = NearestNeighbors(n_neighbors=1, algorithm='auto', metric=self.metric)
        self.nn_model.fit(self.visual_words)
        
        return self
        
    def compute_bow(self, descriptors):
        """
        Compute the Bag of Words representation for a single image
        
        Args:
            descriptors: Feature descriptors of the image, shape (n, d)
            
        Returns:
            hist: Normalized BoW vector (word frequency histogram)
        """
        if self.nn_model is None:
            if self.visual_words is None:
                raise ValueError("Visual dictionary not loaded, cannot build nearest neighbor model")
            self.fit(verbose=False)
            
        if descriptors.shape[0] == 0:
            # If no descriptors, return zero vector
            return np.zeros(len(self.visual_words))
            
        # Normalize descriptors (if the visual dictionary was built with normalized data)
        if self.scaler is not None:
            descriptors = self.scaler.transform(descriptors)
            
        # Find the closest visual word for each descriptor
        distances, indices = self.nn_model.kneighbors(descriptors)
        
        # Build word frequency histogram
        hist = np.zeros(len(self.visual_words))
        for idx in indices.flatten():
            hist[idx] += 1
            
        # L2 normalization
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
            
        return hist
        
    def compute_image_bows(self, features_dict, save_path=None, verbose=True):
        """
        Compute Bag of Words representations for multiple images
        
        Args:
            features_dict: Dictionary of image features, {image_name: (keypoints, descriptors)}
            save_path: If provided, save results to this path
            verbose: Whether to show progress information
            
        Returns:
            images_bow: Dictionary of BoW representations, {image_name: bow_vector}
        """
        if self.nn_model is None:
            if self.visual_words is None:
                raise ValueError("Visual dictionary not loaded, cannot build nearest neighbor model")
            self.fit(verbose=verbose)
            
        self.images_bow = {}
        
        # Use tqdm to show progress bar
        iterator = tqdm(features_dict.items()) if verbose else features_dict.items()
        
        for image_name, (_, descriptors) in iterator:
            if verbose:
                iterator.set_description(f"Computing BoW representation for {image_name}")
                
            if descriptors is not None and descriptors.shape[0] > 0:
                bow = self.compute_bow(descriptors)
                self.images_bow[image_name] = bow
                
        if verbose:
            print(f"Computed BoW representations for {len(self.images_bow)} images")
            
        # Save results to file
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'images_bow': self.images_bow,
                    'metric': self.metric,
                    'visual_words': self.visual_words,
                    'scaler': self.scaler
                }, f)
            print(f"BoW representations saved to {save_path}")
            
        return self.images_bow
        
    def save(self, filepath):
        """
        Save the BoW model to file
        
        Args:
            filepath: Save path
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'images_bow': self.images_bow,
                'metric': self.metric,
                'visual_words': self.visual_words,
                'scaler': self.scaler
            }, f)
            
        print(f"BoW model saved to {filepath}")
        
    @classmethod
    def load(cls, filepath, dictionary_path=None):
        """
        Load a BoW model from file
        
        Args:
            filepath: Path to the BoW model file
            dictionary_path: Path to the visual dictionary file, if needed to recompute BoW representations
            
        Returns:
            BagOfWords: Loaded BoW model instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        # Get the metric
        metric = data.get('metric', 'euclidean')
        
        # Create a new instance
        bow = cls(metric=metric)
        
        # Load BoW representations
        bow.images_bow = data.get('images_bow', {})
        
        # Load visual dictionary information
        bow.visual_words = data.get('visual_words', None)
        bow.scaler = data.get('scaler', None)
        
        # If visual words are not in the saved file but a dictionary path is provided, load from there
        if bow.visual_words is None and dictionary_path is not None:
            try:
                bow._load_dictionary_from_file(dictionary_path)
                print(f"Loaded visual dictionary from {dictionary_path}")
            except Exception as e:
                print(f"Error loading visual dictionary from {dictionary_path}: {e}")
        
        # Initialize the nearest neighbor model
        if bow.visual_words is not None:
            bow.fit(verbose=False)
            
        return bow
        
    def search(self, query_bow, top_k=10):
        """
        Search for the most similar images using BoW representation
        
        Args:
            query_bow: BoW representation of the query image
            top_k: Number of most similar images to return
            
        Returns:
            top_matches: List of most similar images, [(image_name, similarity_score), ...]
        """
        if not self.images_bow:
            raise ValueError("Database is empty, cannot perform search")
            
        similarities = {}
        
        for image_name, bow in self.images_bow.items():
            # Calculate cosine similarity (1 - cosine distance)
            sim = 1 - cosine(query_bow, bow)
            similarities[image_name] = sim
            
        # Sort by similarity in descending order
        sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return sorted_matches[:top_k]

# Example usage
if __name__ == "__main__":
    import feature_extractor
    import visual_dictionary
    
    # Paths
    features_path = "features/sift_features.pkl"
    dictionary_path = "features/visual_dictionary_sift.pkl"
    bow_path = "features/bow_model_sift.pkl"
    
    # Load features
    if os.path.exists(features_path):
        print(f"Loading existing features from {features_path}")
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
    else:
        print(f"Feature file not found: {features_path}")
        exit(1)
    
    # Load or build visual dictionary
    if os.path.exists(dictionary_path):
        print(f"Loading existing visual dictionary from {dictionary_path}")
        vd = visual_dictionary.VisualDictionary.load(dictionary_path)
    else:
        print(f"Visual dictionary file not found: {dictionary_path}")
        exit(1)
    
    # Create BoW model
    bow_model = BagOfWords(visual_dictionary=vd)
    
    # Compute BoW representations for all images
    bow_model.compute_image_bows(features, save_path=bow_path)
    
    # Select an image as a query example
    query_image_name = list(features.keys())[0]
    query_keypoints, query_descriptors = features[query_image_name]
    
    # Compute BoW representation for the query image
    query_bow = bow_model.compute_bow(query_descriptors)
    
    # Search for the most similar images
    results = bow_model.search(query_bow, top_k=5)
    
    print(f"Query: {query_image_name}")
    print("Results:")
    for i, (img_name, score) in enumerate(results, 1):
        print(f"  {i}. {img_name} (Score: {score:.4f})") 