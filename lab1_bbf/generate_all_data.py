#!/usr/bin/env python3
import os
import subprocess
import argparse
import sys
import time

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Generate data files for BBF algorithm testing with different dimensions.')
    parser.add_argument('--dimensions', type=int, nargs='+', default=[2, 4, 8, 16],
                        help='List of dimensions to generate data for (default: [2, 4, 8, 16])')
    parser.add_argument('--files-per-dim', type=int, default=5,
                        help='Number of files to generate for each dimension (default: 5)')
    parser.add_argument('--n-points', type=int, default=100000,
                        help='Number of data points per file (default: 100000)')
    parser.add_argument('--m-queries', type=int, default=100,
                        help='Number of query points per file (default: 100)')
    parser.add_argument('--prefix', type=str, default='data',
                        help='Prefix for generated files (default: "data")')
    parser.add_argument('--compile-only', action='store_true',
                        help='Only compile the C++ generator, do not generate data')
    parser.add_argument('--no-compile', action='store_true',
                        help='Skip compilation, just run the generator')
    
    args = parser.parse_args()
    
    # 确保data目录存在
    os.makedirs('data', exist_ok=True)
    
    # 编译C++程序
    if not args.no_compile:
        print("Compiling generateData.cpp...")
        compile_cmd = ['g++', 'generateData.cpp', '-o', 'generateData']
        try:
            subprocess.run(compile_cmd, check=True)
            print("Compilation successful.")
        except subprocess.CalledProcessError as e:
            print(f"Error during compilation: {e}")
            return 1
        except FileNotFoundError:
            print("Error: g++ compiler not found. Please install g++ or ensure it's in your PATH.")
            return 1
    
    if args.compile_only:
        print("Compilation only mode, exiting without generating data.")
        return 0
    
    # 生成不同维度的数据
    for d in args.dimensions:
        print(f"\n{'-'*50}")
        print(f"Generating data for dimension {d}...")
        
        cmd = [
            './generateData',
            '-d', str(d),
            '-T', str(args.files_per_dim),
            '-n', str(args.n_points),
            '-m', str(args.m_queries),
            '-prefix', args.prefix
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        try:
            start_time = time.time()
            subprocess.run(cmd, check=True)
            end_time = time.time()
            print(f"Generated data for dimension {d} in {end_time - start_time:.2f} seconds.")
        except subprocess.CalledProcessError as e:
            print(f"Error generating data for dimension {d}: {e}")
        except FileNotFoundError:
            print("Error: generateData executable not found. Did compilation fail?")
            return 1
    
    print(f"\n{'-'*50}")
    print(f"Data generation complete for dimensions: {args.dimensions}")
    print(f"Each dimension has {args.files_per_dim} files with {args.n_points} data points and {args.m_queries} query points.")
    print(f"Files are stored in the 'data/' directory with pattern: {args.prefix}_<dim>D_<num>.txt")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 