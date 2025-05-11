# Image Retrieval System Based on Hierarchical Clustering

## Overview

This project implements an image retrieval system based on hierarchical clustering and the Bag of Words (BoW) model. The system extracts SIFT features from images, constructs a visual dictionary using hierarchical clustering, represents images as word frequency histograms, and retrieves similar images based on cosine similarity.

## Features

- **SIFT Feature Extraction**: Extracts scale-invariant feature descriptors from images
- **Hierarchical Clustering**: Builds a visual dictionary using agglomerative clustering
- **Bag of Words Representation**: Represents images as histograms of visual word frequencies
- **Efficient Retrieval**: Performs fast image retrieval based on cosine similarity
- **Comprehensive Evaluation**: Tools for performance analysis and parameter tuning

## Requirements

- Python 3.6+
- OpenCV (`opencv-python`)
- NumPy
- scikit-learn
- SciPy
- Matplotlib
- tqdm

Install all dependencies with:

```bash
pip install numpy opencv-python scikit-learn matplotlib scipy tqdm
```

## Project Structure

- `feature_extractor.py`: SIFT feature extraction module
- `visual_dictionary.py`: Visual dictionary construction using hierarchical clustering
- `bow_representation.py`: Bag of Words representation and retrieval
- `image_retrieval_system.py`: Main system integrating all components
- `run_experiments_part2.py`: Experiment scripts for performance evaluation
- `query_examples.py`: Example script for image querying
- `report_part2.md`: Detailed experimental report
- `photo/`: Directory containing the image dataset

## Usage

### Basic Experiment

Run a basic experiment with default parameters:

```bash
python run_experiments_part2.py --basic
```

### Compare Vocabulary Sizes

Test the system with different vocabulary sizes:

```bash
python run_experiments_part2.py --vocab-sizes --vocab-size-range "50,100,200,400"
```

### Run All Experiments and Save Results

Run all experiments and save results to CSV:

```bash
python run_experiments_part2.py --run-all --save-csv
```

### Query Examples

Run example queries with random images from the dataset:

```bash
python query_examples.py
```

### Custom Query

```python
from image_retrieval_system import ImageRetrievalSystem

# Create system
system = ImageRetrievalSystem(vocab_size=200)

# Setup system
system.setup(data_dir="photo", max_features=500)

# Query with an image
results = system.query_image_path("path/to/image.jpg", top_k=5)
for i, (img_path, score) in enumerate(results, 1):
    print(f"{i}. {img_path} (Score: {score:.4f})")
```

## Performance

The system performs efficiently with the following typical metrics:
- Setup time: ~5-6 seconds for 500 images
- Query time: ~15-20ms per query
- Memory usage: Depends on the dataset size and vocabulary size

## Parameter Tuning

Key parameters that affect system performance:
- `vocab_size`: Size of the visual dictionary (default: 200)
- `max_features`: Maximum number of features extracted per image (default: 500)
- `linkage`: Linkage criteria for hierarchical clustering (default: 'ward')
- `affinity`: Distance metric for clustering (default: 'euclidean')

## License

This project is released under the MIT License.

## Acknowledgements

This project was developed as part of the computer vision and image processing course. The implementation is based on the algorithm described in the paper "Video Google: A Text Retrieval Approach to Object Matching in Videos" by Sivic and Zisserman. 