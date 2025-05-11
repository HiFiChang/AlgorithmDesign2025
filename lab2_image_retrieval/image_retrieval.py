#!/usr/bin/env python3
'''
Image Retrieval System Main Script

This script provides a unified command-line interface to run different image retrieval systems:
1. Improved SIFT feature-based image retrieval system
2. Multi-feature fusion image retrieval system (combining SIFT and color features)

Usage examples:
    # Run the improved SIFT feature retrieval system
    python image_retrieval.py --system improved --query photo/1.png
    
    # Run the multi-feature fusion retrieval system
    python image_retrieval.py --system multi --query photo/1.png --sift_weight 0.6 --color_weight 0.4
'''
import os
import sys
import argparse
import time
import random

# Add current directory to path to ensure modules in subdirectories can be imported
sys.path.append('.')

# Import systems
from systems.improved_retrieval_system import ImprovedRetrievalSystem
from systems.multi_feature_retrieval_system import MultiFeatureRetrievalSystem

def main():
    '''Main function'''
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Unified Image Retrieval System Script")
    
    # System selection
    parser.add_argument("--system", type=str, choices=["improved", "multi"], default="multi",
                      help="System type to run: improved=Improved SIFT system, multi=Multi-feature fusion system")
    
    # Data and query parameters
    parser.add_argument("--data_dir", type=str, default="photo", help="Image directory")
    parser.add_argument("--query", type=str, help="Path to query image")
    parser.add_argument("--random_queries", type=int, default=0, 
                      help="Number of random queries to execute; if set, ignores --query parameter")
    
    # General system parameters
    parser.add_argument("--force_recompute", action="store_true", help="Force recomputation of features and dictionary")
    parser.add_argument("--vocab_size", type=int, default=400, help="SIFT dictionary size")
    parser.add_argument("--max_features", type=int, default=1000, help="Maximum number of feature points per image")
    parser.add_argument("--sim_method", type=str, default="rerank", 
                      choices=["cosine", "euclidean", "combined", "rerank"],
                      help="Similarity calculation method")
    
    # Multi-feature system specific parameters
    parser.add_argument("--color_bins", type=int, default=16, help="Number of bins per channel for color histogram")
    parser.add_argument("--color_space", type=str, default="hsv", choices=["rgb", "hsv"], 
                      help="Color space")
    parser.add_argument("--sift_weight", type=float, default=0.7, help="SIFT feature weight")
    parser.add_argument("--color_weight", type=float, default=0.3, help="Color feature weight")
    
    args = parser.parse_args()
    
    # Create results directory
    if args.system == "improved":
        results_dir = "improved_results"
    else:
        results_dir = "multi_feature_results"
    
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"=== Running {args.system} image retrieval system ===")
    
    # Create and setup system
    system = None
    if args.system == "improved":
        print(f"Creating improved SIFT system (dictionary size: {args.vocab_size})")
        system = ImprovedRetrievalSystem(vocab_size=args.vocab_size)
    else:
        print(f"Creating multi-feature fusion system (SIFT dictionary size: {args.vocab_size}, "
              f"Color bins: {args.color_bins}, Color space: {args.color_space})")
        system = MultiFeatureRetrievalSystem(
            vocab_size=args.vocab_size,
            color_bins=args.color_bins,
            color_space=args.color_space
        )
        # Set feature weights
        system.sift_weight = args.sift_weight
        system.color_weight = args.color_weight
        print(f"SIFT feature weight: {args.sift_weight}, Color feature weight: {args.color_weight}")
    
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
    
    # If random query count is specified
    if args.random_queries > 0:
        # Select random query images
        image_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"Error: No images found in {args.data_dir}")
            return
        
        # Randomly select query images
        random.seed(42)  # Ensure reproducibility
        query_images = random.sample(image_files, min(args.random_queries, len(image_files)))
        
        # Execute queries
        success_count = 0
        for i, query_path in enumerate(query_images):
            print(f"\n===== Random query {i+1}: {os.path.basename(query_path)} =====")
            
            try:
                # Execute query
                results = system.query_image_path(query_path, sim_method=args.sim_method)
                
                # Display results
                save_path = f"{results_dir}/random_query_{i+1}_{os.path.basename(query_path)}.png"
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
        success_rate = success_count / len(query_images) * 100
        print(f"\nOverall results: {success_count}/{len(query_images)} query images ranked first ({success_rate:.1f}%)")
        
        if success_rate == 100:
            print("✓ System optimization successful, all query images ranked first!")
        else:
            print("! System still needs optimization, some query images not ranked first.")
    
    # If a specific query image is specified
    elif args.query:
        if os.path.exists(args.query):
            print(f"\nExecuting specific query: {args.query}")
            results = system.query_image_path(args.query, sim_method=args.sim_method)
            
            # Display results
            save_path = f"{results_dir}/specific_query_{os.path.basename(args.query)}.png"
            system.display_query_results(args.query, results, save_path=save_path)
            
            # Analyze query image ranking
            query_rank = -1
            for i, (path, _) in enumerate(results):
                if path == args.query:
                    query_rank = i
                    break
            
            if query_rank == 0:
                print(f"✓ Success: Query image {os.path.basename(args.query)} ranked in first position")
            elif query_rank > 0:
                print(f"✗ Failure: Query image {os.path.basename(args.query)} ranked at position {query_rank+1}")
            else:
                print(f"✗ Failure: Query image {os.path.basename(args.query)} not found in results")
        else:
            print(f"Error: Query image {args.query} does not exist")
    else:
        print("No query image provided. Use --query parameter to specify query image or --random_queries to execute random queries.")

if __name__ == "__main__":
    main()
