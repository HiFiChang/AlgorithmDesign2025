#!/usr/bin/env python3
import os
import sys
import time
import argparse
import random
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt errors
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append('.')

from systems.improved_retrieval_system import ImprovedRetrievalSystem

def main():
    """Test the improved image retrieval system, ensuring query image ranks first"""
    # Command line arguments
    parser = argparse.ArgumentParser(description="Test the improved image retrieval system")
    parser.add_argument("--data_dir", type=str, default="photo", help="Image directory")
    parser.add_argument("--vocab_size", type=int, default=400, help="Dictionary size")
    parser.add_argument("--max_features", type=int, default=1000, help="Maximum features per image")
    parser.add_argument("--force_recompute", action="store_true", help="Force recomputation of features and dictionary")
    parser.add_argument("--num_queries", type=int, default=5, help="Number of random query samples")
    parser.add_argument("--sim_method", type=str, default="rerank", 
                      choices=["cosine", "euclidean", "combined", "rerank"],
                      help="Similarity calculation method")
    parser.add_argument("--query_image", type=str, help="Path to specific query image")
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs("improved_results", exist_ok=True)
    
    print("=== Testing improved image retrieval system (ensuring self-match ranks first) ===")
    
    # Create and setup system
    print(f"Creating system (dictionary size: {args.vocab_size}, similarity method: {args.sim_method})")
    system = ImprovedRetrievalSystem(vocab_size=args.vocab_size)
    
    # Setup system
    setup_start = time.time()
    success = system.setup(
        data_dir=args.data_dir,
        max_features=args.max_features,
        force_recompute=args.force_recompute
    )
    setup_time = time.time() - setup_start
    
    if not success:
        print("System setup failed")
        return
    
    print(f"System setup completed in {setup_time:.2f} seconds")
    
    # If a specific query image is provided
    if args.query_image:
        if os.path.exists(args.query_image):
            print(f"\nExecuting specific query: {args.query_image}")
            results = system.query_image_path(args.query_image, sim_method=args.sim_method)
            
            # Display results
            save_path = f"improved_results/specific_query_{os.path.basename(args.query_image)}.png"
            system.display_query_results(args.query_image, results, save_path=save_path)
            
            # Analyze query image ranking
            query_rank = -1
            for i, (path, _) in enumerate(results):
                if path == args.query_image:
                    query_rank = i
                    break
            
            if query_rank == 0:
                print(f"✓ Success: Query image {os.path.basename(args.query_image)} ranked in first position")
            elif query_rank > 0:
                print(f"✗ Failure: Query image {os.path.basename(args.query_image)} ranked at position {query_rank+1}")
            else:
                print(f"✗ Failure: Query image {os.path.basename(args.query_image)} not found in results")
            
            return
    
    # Select random query images
    image_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Error: No images found in {args.data_dir}")
        return
    
    # Randomly select query images
    random.seed(42)  # Ensure reproducibility
    query_images = random.sample(image_files, min(args.num_queries, len(image_files)))
    
    # Execute queries
    success_count = 0
    for i, query_path in enumerate(query_images):
        print(f"\n===== Random query {i+1}: {os.path.basename(query_path)} =====")
        
        try:
            # Execute query
            results = system.query_image_path(query_path, sim_method=args.sim_method)
            
            # Display results
            save_path = f"improved_results/random_query_{i+1}_{os.path.basename(query_path)}.png"
            system.display_query_results(query_path, results, save_path=save_path)
            
            # Analyze query image ranking
            query_rank = -1
            for j, (path, _) in enumerate(results):
                if path == query_path:
                    query_rank = j
                    break
            
            if query_rank == 0:
                print(f"✓ Success: Query image ranked in first position")
                success_count += 1
            elif query_rank > 0:
                print(f"✗ Failure: Query image ranked at position {query_rank+1}")
            else:
                print(f"✗ Failure: Query image not found in results")
                
        except Exception as e:
            print(f"Query error: {e}")
    
    # Print overall results
    if args.num_queries > 0:
        success_rate = success_count / len(query_images) * 100
        print(f"\nOverall results: {success_count}/{len(query_images)} query images ranked first ({success_rate:.1f}%)")
        
        if success_rate == 100:
            print("✓ System optimization successful, all query images ranked first!")
        else:
            print("! System still needs optimization, some query images not ranked first.")

if __name__ == "__main__":
    main() 