#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
import pickle

# Set matplotlib backend to 'Agg' to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import csv
import random

from feature_extractor import FeatureExtractor
from visual_dictionary import VisualDictionary
from bow_representation import BagOfWords
from image_retrieval_system import ImageRetrievalSystem

def setup_directories():
    """Create necessary directories if they don't exist."""
    if not os.path.exists('features'):
        os.makedirs('features')
    if not os.path.exists('results'):
        os.makedirs('results')

def run_basic_experiment(data_dir="photo", query_samples=10, vocab_size=200, force_recompute=False, 
                        top_k=10, linkage='ward', affinity='euclidean'):
    """Run a basic experiment using SIFT features with default parameters."""
    print(f"\n--- [1/2] Running Basic SIFT Experiment ---")
    system = ImageRetrievalSystem(vocab_size=vocab_size, 
                                linkage=linkage, affinity=affinity)
    
    # Start setup timer
    setup_start = time.time()
    system.setup(data_dir=data_dir, max_features=500, force_recompute=force_recompute)
    setup_time = time.time() - setup_start
    
    # Skip if the system is not fully set up
    if not system.is_ready():
        print(f"Error: System not fully set up or BoW model unavailable.")
        return {
            "feature_method": "sift",
            "vocab_size": vocab_size,
            "status": "failed",
            "setup_time": setup_time,
            "avg_query_time": 0,
            "num_images": 0,
            "linkage": linkage,
            "affinity": affinity
        }
    
    # Get a list of image paths
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Select random query images
    if query_samples > len(image_paths):
        query_samples = len(image_paths)
    
    np.random.seed(42)  # For reproducibility
    query_indices = np.random.choice(len(image_paths), query_samples, replace=False)
    query_paths = [image_paths[i] for i in query_indices]
    
    # Measure query time
    total_query_time = 0
    query_count = 0
    
    print(f"Running {query_samples} query sample(s)...")
    for query_path in tqdm(query_paths):
        try:
            query_start = time.time()
            results = system.query_image_path(query_path, top_k=top_k)
            query_time = time.time() - query_start
            total_query_time += query_time
            query_count += 1
            
            # Just print the first query for illustration
            if query_count == 1:
                print(f"\nSample query: {os.path.basename(query_path)}")
                for i, (img_path, score) in enumerate(results, 1):
                    print(f"  {i}. {os.path.basename(img_path)} (similarity: {score:.4f})")
        except Exception as e:
            print(f"Error during query: {e}")
    
    if query_count > 0:
        avg_query_time = total_query_time / query_count
        print(f"\nResults for SIFT (vocab size {vocab_size}):")
        print(f"  Average query time: {avg_query_time:.4f} seconds")
        print(f"  Setup time: {setup_time:.2f} seconds")
        print(f"  Number of processed images: {len(image_paths)}")
        print(f"  Feature method: SIFT")
        
        return {
            "feature_method": "sift",
            "vocab_size": vocab_size,
            "status": "success",
            "setup_time": setup_time,
            "avg_query_time": avg_query_time,
            "num_images": len(image_paths),
            "linkage": linkage,
            "affinity": affinity
        }
    else:
        print("Error: No successful queries conducted.")
        return {
            "feature_method": "sift",
            "vocab_size": vocab_size,
            "status": "failed",
            "setup_time": setup_time,
            "avg_query_time": 0,
            "num_images": len(image_paths),
            "linkage": linkage,
            "affinity": affinity
        }

def compare_vocab_sizes(feature_method='sift', max_features=500, data_dir="photo", 
                         query_samples=10, vocab_sizes=[50, 100, 200, 400], 
                         force_recompute=False, top_k=10, linkage='ward', affinity='euclidean'):
    """
    Compare performance with different vocabulary sizes.
    
    Args:
        feature_method: Feature extraction method (sift)
        max_features: Maximum number of features to extract per image
        data_dir: Directory containing the image dataset
        query_samples: Number of random images to use for querying
        vocab_sizes: List of vocabulary sizes to test
        force_recompute: Force recomputation of features and dictionary
        top_k: Number of top results to return
        linkage: Hierarchical clustering linkage method
        affinity: Distance metric for clustering and similarity
        
    Returns:
        Dictionary of results for each vocabulary size
    """
    print(f"\n--- [2/2] Running Vocabulary Size Comparison (using SIFT) ---")
    results_by_vocab = {}
    successful_runs = 0
    
    for vocab_size in sorted(vocab_sizes):
        print(f"\nTesting vocabulary size: {vocab_size}")
        system = ImageRetrievalSystem(vocab_size=vocab_size,
                                    linkage=linkage, affinity=affinity)
        
        # Setup timer
        setup_start = time.time()
        system.setup(data_dir=data_dir, max_features=max_features, force_recompute=force_recompute)
        setup_time = time.time() - setup_start
        
        # Skip if the system is not fully set up
        if not system.is_ready():
            print(f"Error: System (vocab size {vocab_size}) not fully set up.")
            results_by_vocab[vocab_size] = {
                "feature_method": feature_method,
                "vocab_size": vocab_size,
                "status": "failed",
                "setup_time": setup_time,
                "avg_query_time": 0,
                "linkage": linkage,
                "affinity": affinity
            }
            continue
        
        # Get a list of image paths
        image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Select random query images
        if query_samples > len(image_paths):
            query_samples = len(image_paths)
        
        np.random.seed(42)  # For reproducibility
        query_indices = np.random.choice(len(image_paths), query_samples, replace=False)
        query_paths = [image_paths[i] for i in query_indices]
        
        # Measure query time
        total_query_time = 0
        query_count = 0
        
        print(f"Running {query_samples} query sample(s)...")
        for query_path in tqdm(query_paths):
            try:
                query_start = time.time()
                results = system.query_image_path(query_path, top_k=top_k)
                query_time = time.time() - query_start
                total_query_time += query_time
                query_count += 1
            except Exception as e:
                print(f"Error during query with vocab size {vocab_size}: {e}")
        
        if query_count > 0:
            avg_query_time = total_query_time / query_count
            print(f"Results for vocab size {vocab_size}:")
            print(f"  Average query time: {avg_query_time:.4f} seconds")
            print(f"  Setup time: {setup_time:.2f} seconds")
            
            results_by_vocab[vocab_size] = {
                "feature_method": feature_method,
                "vocab_size": vocab_size,
                "status": "success",
                "setup_time": setup_time,
                "avg_query_time": avg_query_time,
                "linkage": linkage,
                "affinity": affinity
            }
            successful_runs += 1
        else:
            print(f"Error: No successful queries for vocab size {vocab_size}.")
            results_by_vocab[vocab_size] = {
                "feature_method": feature_method,
                "vocab_size": vocab_size,
                "status": "failed",
                "setup_time": setup_time,
                "avg_query_time": 0,
                "linkage": linkage,
                "affinity": affinity
            }
    
    print("\nVocabulary Size Comparison Results:")
    for vocab_size in sorted(results_by_vocab.keys()):
        result = results_by_vocab[vocab_size]
        if result['status'] == "success":
            print(f"  Vocab Size {vocab_size}: {result['avg_query_time']:.4f} seconds (setup: {result['setup_time']:.2f} seconds)")
        else:
            print(f"  Vocab Size {vocab_size}: Failed")
    
    if successful_runs > 1: # Need at least two successful runs to plot
        plt.figure(figsize=(12, 6))
        valid_vocab_sizes = [s for s, r in results_by_vocab.items() if r['status'] == "success" and r.get('avg_query_time', 0) > 0]
        
        if len(valid_vocab_sizes) > 1: # Ensure enough data points for line plot
            plt.subplot(1, 2, 1)
            query_times_vocab = [results_by_vocab[s]['avg_query_time'] for s in valid_vocab_sizes]
            plt.plot(valid_vocab_sizes, query_times_vocab, marker='o', linestyle='-', color='dodgerblue')
            plt.xlabel('Vocabulary Size')
            plt.ylabel('Average Query Time (seconds)')
            plt.title(f'Impact of Vocabulary Size on Query Time (SIFT)')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            setup_times_vocab = [results_by_vocab[s]['setup_time'] for s in valid_vocab_sizes]
            plt.plot(valid_vocab_sizes, setup_times_vocab, marker='s', linestyle='--', color='coral')
            plt.xlabel('Vocabulary Size')
            plt.ylabel('System Setup Time (seconds)')
            plt.title(f'Impact of Vocabulary Size on Setup Time (SIFT)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'vocab_size_comparison_sift.png')
            print(f"Comparison chart saved to vocab_size_comparison_sift.png")
        else:
            print(f"Not enough successful vocabulary size experiments ({len(valid_vocab_sizes)}) to generate comparison chart.")
        # plt.show() # Using Agg backend
    else:
        print("Not enough successful experiments to generate vocabulary size comparison chart.")
    
    return results_by_vocab

def save_results_to_csv(all_results, filename="image_retrieval_results.csv"):
    """Save all experiment results to a CSV file."""
    # Flatten results from nested dictionaries
    flat_results = []
    
    # Basic experiment
    if 'basic' in all_results:
        flat_results.append(all_results['basic'])
    
    # Vocab size comparison
    if 'vocab_size' in all_results:
        for vocab_size, result in all_results['vocab_size'].items():
            flat_results.append(result)
    
    # If there are no results, don't create an empty file
    if not flat_results:
        print("No results to save.")
        return
    
    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        # Get all possible fields from the results
        fieldnames = set()
        for result in flat_results:
            fieldnames.update(result.keys())
        fieldnames = sorted(list(fieldnames))
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in flat_results:
            writer.writerow(result)
    
    print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Run image retrieval experiments")
    parser.add_argument("--run-all", action="store_true", help="Run all experiments")
    parser.add_argument("--basic", action="store_true", help="Run basic SIFT experiment")
    parser.add_argument("--vocab-sizes", action="store_true", help="Run vocabulary size comparison")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputation of features and dictionaries")
    parser.add_argument("--save-csv", action="store_true", help="Save results to CSV file")
    parser.add_argument("--data-dir", type=str, default="photo", help="Directory containing the image dataset")
    parser.add_argument("--query-samples", type=int, default=10, help="Number of random images to use for querying")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top results to return")
    parser.add_argument("--max-features", type=int, default=500, help="Maximum number of features to extract per image")
    parser.add_argument("--vocab-size", type=int, default=200, help="Vocabulary size for basic experiment")
    parser.add_argument("--vocab-size-range", type=str, default="50,100,200,400", 
                        help="Comma-separated list of vocabulary sizes to test")
    parser.add_argument("--linkage", type=str, default="ward", 
                        help="Hierarchical clustering linkage method (ward, complete, average, single)")
    parser.add_argument("--affinity", type=str, default="euclidean", 
                        help="Distance metric for clustering and similarity (euclidean, cosine, etc.)")
    
    args = parser.parse_args()
    
    # Create necessary directories
    setup_directories()
    
    # Parse vocabulary sizes for comparison
    vocab_sizes = [int(vs) for vs in args.vocab_size_range.split(',')]
    
    # Initialize results dictionary
    all_results = {}
    
    # Run experiments based on arguments
    if args.run_all or args.basic:
        basic_result = run_basic_experiment(
            data_dir=args.data_dir,
            query_samples=args.query_samples,
            vocab_size=args.vocab_size,
            force_recompute=args.force_recompute,
            top_k=args.top_k,
            linkage=args.linkage,
            affinity=args.affinity
        )
        all_results['basic'] = basic_result
    
    if args.run_all or args.vocab_sizes:
        vocab_results = compare_vocab_sizes(
            feature_method='sift',
            max_features=args.max_features,
            data_dir=args.data_dir,
            query_samples=args.query_samples,
            vocab_sizes=vocab_sizes,
            force_recompute=args.force_recompute,
            top_k=args.top_k,
            linkage=args.linkage,
            affinity=args.affinity
        )
        all_results['vocab_size'] = vocab_results
    
    # Save results to CSV if requested
    if args.save_csv:
        save_results_to_csv(all_results)
    
    print("\nAll experiments completed!")

if __name__ == "__main__":
    main() 