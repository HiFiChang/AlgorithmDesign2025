#!/usr/bin/env python3
import os
import numpy as np
import cv2
import pickle
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from feature_extractor import FeatureExtractor
from visual_dictionary import VisualDictionary
from bow_representation import BagOfWords

class ImageRetrievalSystem:
    """
    Image Retrieval System that uses SIFT features and hierarchical clustering
    for visual dictionary construction.
    """
    
    def __init__(self, vocab_size=200, linkage='ward', affinity='euclidean'):
        """
        Initialize the image retrieval system.
        
        Args:
            vocab_size: Size of the visual dictionary (number of visual words)
            linkage: Linkage criteria for hierarchical clustering ('ward', 'complete', 'average', 'single')
            affinity: Distance metric for clustering and similarity calculations
        """
        self.vocab_size = vocab_size
        self.linkage = linkage
        self.affinity = affinity
        
        # Initialize feature extractor (SIFT)
        self.feature_extractor = None
        
        # Paths for saving intermediate files
        self.features_path = f"features/sift_features.pkl"
        self.dictionary_path = f"features/visual_dictionary_sift_{vocab_size}.pkl"
        self.bow_model_path = f"features/bow_model_sift_{vocab_size}.pkl"
        
        # Data structures
        self.features = {}  # {image_path: (keypoints, descriptors)}
        self.visual_dictionary = None
        self.bow_model = None
        
        # Status flags
        self._features_loaded = False
        self._dictionary_built = False
        self._bow_computed = False
    
    def setup(self, data_dir="photo", max_features=500, force_recompute=False):
        """
        Setup the image retrieval system by extracting features, building dictionary,
        and computing BoW representations.
        
        Args:
            data_dir: Directory containing the images
            max_features: Maximum number of features to extract per image
            force_recompute: Force recomputation of all components
            
        Returns:
            bool: True if setup was successful, False otherwise
        """
        print("Setting up Image Retrieval System...")
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(max_features=max_features)
        
        # Extract features from images
        if not self._extract_features_from_directory(data_dir, force_recompute):
            print("Error: Failed to extract or load features")
            return False
        
        # Build visual dictionary
        if not self._build_dictionary(force_recompute):
            print("Error: Failed to build or load visual dictionary")
            return False
        
        # Compute BoW representations
        if not self._compute_bow_representations(force_recompute):
            print("Error: Failed to compute or load BoW representations")
            return False
        
        print("Image Retrieval System setup completed successfully")
        return True
    
    def is_ready(self):
        """
        Check if the system is fully set up and ready for queries.
        
        Returns:
            bool: True if the system is ready, False otherwise
        """
        return self._features_loaded and self._dictionary_built and self._bow_computed
    
    def _extract_features_from_directory(self, directory, force_recompute=False):
        """
        Extract SIFT features from all images in the directory.
        
        Args:
            directory: Directory containing images
            force_recompute: Force recomputation of features
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Extracting SIFT features from images in {directory}...")
            start_time = time.time()
            
            # Extract features
            self.features = self.feature_extractor.batch_extract_from_directory(
                directory=directory,
                save_path=self.features_path,
                force_recompute=force_recompute
            )
            
            if not self.features:
                print("Warning: No features extracted or loaded")
                self._features_loaded = False
                return False
            
            self._features_loaded = True
            print(f"Features extracted/loaded in {time.time() - start_time:.2f} seconds")
            return True
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            self._features_loaded = False
            return False
    
    def _build_dictionary(self, force_recompute=False):
        """
        Build a visual dictionary using hierarchical clustering.
        
        Args:
            force_recompute: Force recomputation of the dictionary
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Skip if features haven't been loaded
            if not self._features_loaded:
                print("Error: Features must be loaded before building dictionary")
                return False
            
            # Check if dictionary already exists
            if os.path.exists(self.dictionary_path) and not force_recompute:
                print(f"Loading pre-computed visual dictionary from {self.dictionary_path}")
                try:
                    self.visual_dictionary = VisualDictionary.load(self.dictionary_path)
                    self._dictionary_built = True
                    return True
                except Exception as e:
                    print(f"Error loading dictionary: {e}")
                    print("Will recompute the dictionary")
            
            print(f"Building visual dictionary with {self.vocab_size} words...")
            start_time = time.time()
            
            # Get all descriptors from the features
            all_descriptors = self.feature_extractor.get_all_descriptors(self.features)
            
            if all_descriptors.shape[0] == 0:
                print("Warning: No valid descriptors found to build dictionary. Dictionary building skipped.")
                self._dictionary_built = False
                return False
            
            # Build the dictionary
            self.visual_dictionary = VisualDictionary(
                vocab_size=self.vocab_size,
                linkage=self.linkage,
                affinity=self.affinity
            )
            
            # Fit the dictionary
            self.visual_dictionary.fit(all_descriptors, verbose=True)
            
            # Save the dictionary
            self.visual_dictionary.save(self.dictionary_path)
            
            self._dictionary_built = True
            print(f"Visual dictionary built in {time.time() - start_time:.2f} seconds")
            return True
            
        except Exception as e:
            print(f"Error building visual dictionary: {e}")
            self._dictionary_built = False
            return False
    
    def _compute_bow_representations(self, force_recompute=False):
        """
        Compute Bag-of-Words representations for all images.
        
        Args:
            force_recompute: Force recomputation of BoW representations
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Skip if dictionary hasn't been built
            if not self._dictionary_built:
                print("Error: Visual dictionary must be built before computing BoW representations")
                return False
            
            # Check if BoW model already exists
            if os.path.exists(self.bow_model_path) and not force_recompute:
                print(f"Loading pre-computed BoW model from {self.bow_model_path}")
                try:
                    self.bow_model = BagOfWords.load(self.bow_model_path)
                    self._bow_computed = True
                    return True
                except Exception as e:
                    print(f"Error loading BoW model: {e}")
                    print("Will recompute the BoW model")
            
            print("Computing Bag-of-Words representations...")
            start_time = time.time()
            
            # Create a BoW model
            self.bow_model = BagOfWords(self.visual_dictionary, metric=self.affinity)
            
            # Compute BoW representations for all images
            self.bow_model.compute_image_bows(self.features, save_path=self.bow_model_path)
            
            self._bow_computed = True
            print(f"BoW representations computed in {time.time() - start_time:.2f} seconds")
            return True
            
        except Exception as e:
            print(f"Error computing BoW representations: {e}")
            self._bow_computed = False
            return False
    
    def query_image_path(self, image_path, top_k=10):
        """
        Query the system with an image path to find similar images.
        
        Args:
            image_path: Path to the query image
            top_k: Number of top results to return
            
        Returns:
            list: List of (image_path, similarity_score) tuples for the top-k results
        """
        # Check if system is fully set up
        if not self.is_ready():
            raise ValueError("Error: System not fully set up or BoW model unavailable. Run setup() first.")
        
        # Extract features from the query image
        keypoints, descriptors = self.feature_extractor.extract_from_file(image_path)
        
        if keypoints is None or descriptors is None or len(descriptors) == 0:
            raise ValueError(f"Error: No features could be extracted from {image_path}")
        
        # Convert to BoW representation
        query_bow = self.bow_model.compute_bow(descriptors)
        
        # Search for similar images
        results = self.bow_model.search(query_bow, top_k=top_k)
        
        # Format results
        formatted_results = [(img_path, score) for img_path, score in results]
        
        return formatted_results
    
    def display_query_results(self, query_path, results, max_display=5):
        """
        Display the query image and its top matches.
        
        Args:
            query_path: Path to the query image
            results: List of (image_path, similarity_score) tuples
            max_display: Maximum number of results to display
            
        Returns:
            None
        """
        try:
            import matplotlib.pyplot as plt
            
            # Limit results to max_display
            display_results = results[:min(max_display, len(results))]
            
            # Create figure
            fig = plt.figure(figsize=(12, 4))
            
            # Display query image
            query_img = cv2.imread(query_path)
            query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
            ax = fig.add_subplot(1, len(display_results) + 1, 1)
            ax.imshow(query_img)
            ax.set_title("Query Image")
            ax.axis('off')
            
            # Display result images
            for i, (img_path, score) in enumerate(display_results, 1):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax = fig.add_subplot(1, len(display_results) + 1, i + 1)
                ax.imshow(img)
                ax.set_title(f"Match {i}\nScore: {score:.4f}")
                ax.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error displaying results: {e}")
            print("Results:")
            print(f"Query: {query_path}")
            for i, (img_path, score) in enumerate(results[:max_display], 1):
                print(f"{i}. {img_path} (Score: {score:.4f})")

# Example usage
if __name__ == "__main__":
    # Create an image retrieval system
    system = ImageRetrievalSystem(vocab_size=200, linkage='ward', affinity='euclidean')
    
    # Setup the system
    system.setup(data_dir="photo", max_features=500)
    
    # Query with an image
    if system.is_ready():
        results = system.query_image_path("photo/example.png", top_k=5)
        system.display_query_results("photo/example.png", results) 