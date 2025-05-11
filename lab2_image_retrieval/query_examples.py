#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import cv2
import random
import time
from image_retrieval_system import ImageRetrievalSystem

def display_query_results(query_path, results, save_path=None, title="Query Results"):
    """
    Display query results with images
    
    Args:
        query_path: Path to the query image
        results: List of (image_path, similarity_score) tuples
        save_path: Path to save the result image (if None, display interactively)
        title: Title for the figure
    """
    # Read query image
    query_img = cv2.imread(query_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    n_results = len(results)
    fig, axes = plt.subplots(1, n_results + 1, figsize=(3 * (n_results + 1), 4))
    
    # Display query image
    axes[0].imshow(query_img)
    axes[0].set_title("Query Image")
    axes[0].axis('off')
    
    # Display result images
    for i, (img_path, score) in enumerate(results):
        result_img = cv2.imread(img_path)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        axes[i+1].imshow(result_img)
        axes[i+1].set_title(f"Match {i+1}\nScore: {score:.4f}")
        axes[i+1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Results saved to {save_path}")
    else:
        plt.savefig("query_result.png")
        print("Results saved to query_result.png")
    
    plt.close()

def query_random_images(system, data_dir="photo", num_queries=3, top_k=5):
    """
    Query with random images from the dataset
    
    Args:
        system: ImageRetrievalSystem instance
        data_dir: Directory containing images
        num_queries: Number of random queries to perform
        top_k: Number of top results to return
    """
    # Get all image files
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) == 0:
        print(f"No images found in {data_dir}")
        return
    
    # Select random images
    random.seed(42)  # For reproducibility
    query_images = random.sample(image_files, min(num_queries, len(image_files)))
    
    # Perform queries
    for i, query_img in enumerate(query_images):
        print(f"\nRandom Query {i+1}: {os.path.basename(query_img)}")
        
        try:
            start_time = time.time()
            results = system.query_image_path(query_img, top_k=top_k)
            query_time = time.time() - start_time
            
            print(f"Query time: {query_time:.4f} seconds")
            print("Results:")
            for j, (img_path, score) in enumerate(results, 1):
                print(f"  {j}. {os.path.basename(img_path)} (Score: {score:.4f})")
            
            # Display and save results
            display_query_results(
                query_img, 
                results, 
                save_path=f"results/random_query_{i+1}.png",
                title=f"Random Query {i+1}: {os.path.basename(query_img)}"
            )
            
        except Exception as e:
            print(f"Error querying {query_img}: {e}")

def query_specific_images(system, query_images, top_k=5):
    """
    Query with specific images
    
    Args:
        system: ImageRetrievalSystem instance
        query_images: List of paths to query images
        top_k: Number of top results to return
    """
    for i, query_img in enumerate(query_images):
        if not os.path.exists(query_img):
            print(f"Warning: Image {query_img} does not exist")
            continue
            
        print(f"\nSpecific Query {i+1}: {os.path.basename(query_img)}")
        
        try:
            start_time = time.time()
            results = system.query_image_path(query_img, top_k=top_k)
            query_time = time.time() - start_time
            
            print(f"Query time: {query_time:.4f} seconds")
            print("Results:")
            for j, (img_path, score) in enumerate(results, 1):
                print(f"  {j}. {os.path.basename(img_path)} (Score: {score:.4f})")
            
            # Display and save results
            display_query_results(
                query_img, 
                results, 
                save_path=f"results/specific_query_{i+1}.png",
                title=f"Specific Query {i+1}: {os.path.basename(query_img)}"
            )
            
        except Exception as e:
            print(f"Error querying {query_img}: {e}")

def main():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    print("Loading Image Retrieval System...")
    # Create system with default parameters
    system = ImageRetrievalSystem(vocab_size=200)
    
    # Setup system (will load pre-computed features, dictionary, and BoW model)
    system.setup(data_dir="photo", max_features=500)
    
    # Ensure the system is ready
    if not system.is_ready():
        print("Error: System is not fully set up. Please run with --force-recompute first.")
        return
    
    # Query with random images
    print("\nPerforming random queries...")
    query_random_images(system, data_dir="photo", num_queries=5, top_k=5)
    
    # Let user select their own images to query if desired
    print("\nTo query with specific images, modify the script to add your chosen images.")
    print("Example:")
    print("  query_specific_images(system, [")
    print("      'photo/image1.png',")
    print("      'photo/image2.png',")
    print("      'photo/image3.png'")
    print("  ])")

if __name__ == "__main__":
    main() 