import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before other matplotlib imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set more aesthetic plotting style and fonts
plt.style.use('seaborn-v0_8-pastel')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# Define a nice custom color scheme
COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

# Define result file name and image save directory
CSV_FILE = "bbf_experiment_results.csv"
OUTPUT_DIR = "experiment_plots"

# Create directory to save images (if it doesn't exist)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def plot_path(filename):
    return os.path.join(OUTPUT_DIR, filename)

# Load data
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"Error: {CSV_FILE} not found. Please run run_experiments_part1.py first to generate results.")
    exit()

# Data type conversion (ensure numeric columns are numeric types)
numeric_cols = [
    'kdtree_build_time', 'bruteforce_avg_time', 'kdtree_avg_time',
    'bbf_avg_time', 'kdtree_accuracy', 'bbf_accuracy',
    'bruteforce_memory_mb', 'kdtree_memory_mb', 'bbf_memory_mb'
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Create a single combined chart with 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
fig.suptitle('BBF Algorithm Performance Analysis', fontsize=16, y=0.98)

# 1. Query Time vs Dimension (Top Left)
ax = axs[0, 0]
for i, (algo_prefix, label_name) in enumerate([
    ('bruteforce', 'Brute Force'),
    ('kdtree', 'Standard K-d Tree'),
    ('bbf', 'BBF (t=200)')
]):
    time_col = f'{algo_prefix}_avg_time'
    if time_col in df.columns:
        # Ignore 16D brute force search (since it used k-d tree data)
        plot_df = df.copy()
        if algo_prefix == 'bruteforce':
            plot_df.loc[plot_df['dimension'] == 16, time_col] = float('nan')
        ax.plot(plot_df['dimension'], plot_df[time_col], 'o-', 
                 label=label_name, color=COLORS[i], linewidth=2.5, markersize=8)

ax.set_title('Average Query Time Comparison', fontweight='bold')
ax.set_xlabel('Dimension', fontweight='bold')
ax.set_ylabel('Query Time (seconds)', fontweight='bold')
ax.set_yscale('log')
ax.set_xticks(df['dimension'].unique())
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='best', frameon=True, facecolor='white', edgecolor='lightgray')

# 2. BBF Accuracy vs Dimension (Top Right)
ax = axs[0, 1]
ax.plot(df['dimension'], df['bbf_accuracy'], 'o-', 
         color=COLORS[2], linewidth=2.5, markersize=8, label='BBF (t=200)')
ax.set_title('BBF Accuracy by Dimension', fontweight='bold')
ax.set_xlabel('Dimension', fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_xticks(df['dimension'].unique())
ax.set_ylim(0, 105)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='best', frameon=True, facecolor='white', edgecolor='lightgray')

# Add labels to accuracy data points
for x, y in zip(df['dimension'], df['bbf_accuracy']):
    ax.annotate(f'{y:.1f}%', 
                xy=(x, y), 
                xytext=(0, 10),
                textcoords='offset points',
                ha='center', 
                fontweight='bold',
                fontsize=9)

# 3. Memory Usage vs Dimension (Bottom Left)
ax = axs[1, 0]
for i, (algo_prefix, label_name) in enumerate([
    ('bruteforce', 'Brute Force (Data Size)'),
    ('kdtree', 'Standard K-d Tree (Structure)'),
    ('bbf', 'BBF (t=200)')
]):
    mem_col = f'{algo_prefix}_memory_mb'
    if mem_col in df.columns:
        ax.plot(df['dimension'], df[mem_col], 'o-', 
                 label=label_name, color=COLORS[i], linewidth=2.5, markersize=8)

ax.set_title('Memory Usage Comparison', fontweight='bold')
ax.set_xlabel('Dimension', fontweight='bold')
ax.set_ylabel('Memory Usage (MB)', fontweight='bold')
ax.set_xticks(df['dimension'].unique())
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='best', frameon=True, facecolor='white', edgecolor='lightgray')

# 4. BBF vs Standard K-d Tree Speed Comparison (Bottom Right)
ax = axs[1, 1]
# Calculate speed improvement: Standard k-d time / BBF time
df['bbf_speedup_vs_kdtree'] = df['kdtree_avg_time'] / df['bbf_avg_time']

# Create a bar chart with texture
bars = ax.bar(df['dimension'], df['bbf_speedup_vs_kdtree'], 
               color=COLORS[4], alpha=0.7, width=0.6)

# Add value labels to each bar
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}x',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')

ax.set_title('BBF Speedup Factor vs Standard K-d Tree', fontweight='bold')
ax.set_xlabel('Dimension', fontweight='bold')
ax.set_ylabel('Speedup Factor', fontweight='bold')
ax.axhline(1, color='gray', linestyle='--', alpha=0.7)
ax.set_xticks(df['dimension'].unique())
ax.grid(True, linestyle='--', alpha=0.7)

# Adjust spacing between subplots
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at top for main title

# Add watermark
fig.text(0.5, 0.01, "BBF Algorithm Experiment Analysis", fontsize=10, color='gray', 
         ha='center', va='bottom', alpha=0.5)

# Save the combined chart
plt.savefig(plot_path("bbf_combined_analysis.png"), dpi=300, bbox_inches='tight')
print(f"Combined chart saved to {plot_path('bbf_combined_analysis.png')}")
plt.close()

print(f"\nChart analysis completed and saved in '{OUTPUT_DIR}' directory.")
