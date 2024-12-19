# Motion Prediction for Cars

<img width="839" alt="Screenshot 2024-12-18 at 10 09 06 PM" src="https://github.com/user-attachments/assets/7aa83810-9e28-4c6b-a461-59ce9a7e25a3" />

This repository is a work in progress, I will provides tools for preprocessing the ArgoVerse 2 motion forecasting dataset to prepare it for training motion prediction models. The code focuses on generating appropriate labels by analyzing global changes in position and heading across the dataset and discretizing these changes into meaningful bins. Then passing the data to the models for training/inference.

## Table of Contents
- Soon to come
- [License](#license)
---

## Introduction

Accurate motion prediction is essential for autonomous vehicles to navigate safely by anticipating the future trajectories of surrounding objects. This project provides a framework to preprocess the ArgoVerse 2 dataset, particularly focusing on:

- Transforming positional data into the agent's frame of reference.
- Analyzing global changes in position and heading to determine appropriate bin ranges.
- Generating discrete labels by discretizing continuous changes into bins for classification tasks.
- Running training models
- and More (soon to come)

## Features

- **Data Extraction**: Functions to extract world trajectories and the focal object's trajectory from ArgoVerse 2 `.parquet` files.
- **Coordinate Transformation**: Transforms global positions to the agent's local frame.
- **Global Analysis**: Collects and analyzes global changes in position and heading across the dataset.
- **Label Generation**: Generates labels representing changes in `x`, `y`, and heading, and discretizes them into bins.
- **Visualization**: Tools to visualize data distributions to inform binning strategies.
- **Dataset and DataLoader**: Constructs PyTorch `Dataset` and `DataLoader` objects for model training.
- **More to come**: Training models, running inference, converting from percentages to linear values and more to come.  

---

## Requirements

- Python 3.6 or higher
- Required Python libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `torch` (PyTorch)
  - `pyarrow` or `fastparquet` (for reading `.parquet` files)

Install the required libraries using:


```
pip install pandas numpy matplotlib torch pyarrow # for now 

Usage

Data Preparation
Download the Dataset: Obtain the ArgoVerse 2 motion forecasting dataset from the official website.
Organize Data: Place the .parquet files in a directory structure accessible to your scripts.
Global Change Analysis
Before preprocessing, analyze global changes to determine appropriate bin ranges for discretization.

Update File Paths: Edit the list_of_parquet_files_main list in the collect_global_changes function to include paths to a representative sample of your dataset.
Run Global Analysis: Use the collect_global_changes and analyze_global_distributions functions to collect and analyze global changes.
NOTE: I ran the analysis on the whole of the ArgoVerse 2 dataset

# List of .parquet file paths
list_of_parquet_files_main = [
    # Add file paths as needed, will include scripts for you to do that later on
]

# Collect global changes
global_x_changes, global_y_changes, global_heading_changes = collect_global_changes(list_of_parquet_files_main)

# Analyze global distributions
analyze_global_distributions(global_x_changes, global_y_changes, global_heading_changes)
Determine Bin Ranges: Based on the analysis, set bin ranges for x, y, and heading changes. For example:
x_range_min = 0  # Start from zero if most x changes are positive
x_range_max = 41.7139  # 99th percentile of x changes
y_range_min = -5.5628  # 1st percentile of y changes
y_range_max = 6.2710   # 99th percentile of y changes
heading_range_min = -0.7035  # 1st percentile of heading changes
heading_range_max = 0.7346   # 99th percentile of heading changes
Data Preprocessing
With the bin ranges determined, proceed to preprocess your dataset.

Update the Label Generation Function: Modify generate_discrete_y_labels_global to use the saved bin ranges.
def generate_discrete_y_labels_global(y_labels, bins=50):
    x_changes = y_labels[:, 0]
    y_changes = y_labels[:, 1]
    heading_changes = y_labels[:, 2]
    x_range_min = 0
    x_range_max = 41.7139
    y_range_min = -5.5628
    y_range_max = 6.2710
    heading_range_min = -0.7035
    heading_range_max = 0.7346
    # Clip changes to be within the bin ranges
    x_changes_clipped = np.clip(x_changes, x_range_min, x_range_max)
    y_changes_clipped = np.clip(y_changes, y_range_min, y_range_max)
    heading_changes_clipped = np.clip(heading_changes, heading_range_min, heading_range_max)
    # Discretize the changes
    x_classes, x_bin_edges = discretize_changes_fixed(x_changes_clipped, bins, x_range_min, x_range_max)
    y_classes, y_bin_edges = discretize_changes_fixed(y_changes_clipped, bins, y_range_min, y_range_max)
    heading_classes, heading_bin_edges = discretize_changes_fixed(heading_changes_clipped, bins, heading_range_min, heading_range_max)
    # One-hot encode the classes
    x_one_hot = np.eye(bins)[x_classes]
    y_one_hot = np.eye(bins)[y_classes]
    heading_one_hot = np.eye(bins)[heading_classes]
    # Stack the one-hot encoded classes
    y_discrete_labels = np.stack([x_one_hot, y_one_hot, heading_one_hot], axis=1)
    return y_discrete_labels end ```
    
Process the Dataset: Iterate over your dataset and preprocess each file.
world_trajectories_np = []
focused_trajectories_np = []
labels_np = []

for parquet_path in list_of_parquet_files_main:
    try:
        # Extract trajectories
        world_traj = extract_world_trajectories(parquet_path, max_objects=80, observed_timesteps=80)
        focused_traj = extract_focused_object_trajectory(parquet_path, observed_timesteps=80)

        # Generate labels
        y_labels_continuous = generate_multi_modal_y_labels(parquet_path, future_steps=30)
        y_labels_discrete = generate_discrete_y_labels_global(y_labels_continuous, bins=50)

        # Append data
        world_trajectories_np.append(world_traj)
        focused_trajectories_np.append(focused_traj)
        labels_np.append(y_labels_discrete)
    except Exception as e:
        print(f"Error processing file {parquet_path}: {e}")
Create Dataset and DataLoader:
# Convert lists to numpy arrays
world_trajectories_np = np.array(world_trajectories_np)
focused_trajectories_np = np.array(focused_trajectories_np)
labels_np = np.array(labels_np)

# Create Dataset and DataLoader
dataset = ArgoverseMultiModalDataset(world_trajectories_np, focused_trajectories_np, labels_np)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
Configuration

```
<img width="638" alt="Screenshot 2024-12-18 at 10 13 09 PM" src="https://github.com/user-attachments/assets/f4f52126-1d54-4621-8d50-9a0501709a60" />
<img width="787" alt="Screenshot 2024-12-18 at 10 14 47 PM" src="https://github.com/user-attachments/assets/f06f0370-0541-471d-a10b-9048c39a0086" />

Parameters:
covered_time: Number of observed timesteps (e.g., 80)
future_time: Number of future timesteps to predict (e.g., 30)
max_objects: Maximum number of objects to consider (e.g., 80)
bins: Number of bins for discretization (e.g., 50)
Bin Ranges: Set based on global analysis. Save these ranges for consistent preprocessing.
Contributing

Contributions are welcome! If you have suggestions or encounter issues, please open an issue or submit a pull request.

<img width="1240" alt="Screenshot 2024-11-27 at 11 26 00 PM" src="https://github.com/user-attachments/assets/fe1c155a-ffad-48b7-b844-453d133f6bb1">
cars
￼
X Changes - min: -57.5793, max: 86.5215, mean: 10.1197, std: 10.0379
Y Changes - min: -37.7315, max: 47.2216, mean: -0.0121, std: 1.5879
Heading Changes - min: -3.1391, max: 3.1334, mean: 0.0029, std: 0.186


X Changes Percentiles:
0th percentile: -57.5793
1th percentile: -0.0706
5th percentile: 0.0275
10th percentile: 0.4626
25th percentile: 2.2282
50th percentile: 6.9353
75th percentile: 15.3985
90th percentile: 24.7522
95th percentile: 30.4441
99th percentile: 41.7139
100th percentile: 86.5215

Y Changes Percentiles:
0th percentile: -37.7315
1th percentile: -5.5628
5th percentile: -1.5597
10th percentile: -0.7286
25th percentile: -0.1832
50th percentile: -0.0082
75th percentile: 0.1073
90th percentile: 0.5877
95th percentile: 1.5514
99th percentile: 6.2710
100th percentile: 47.2216

Heading Changes Percentiles:
0th percentile: -3.1391
1th percentile: -0.7035
5th percentile: -0.1936
10th percentile: -0.0676
25th percentile: -0.0138
50th percentile: -0.0000
75th percentile: 0.0135
90th percentile: 0.0759
95th percentile: 0.2275
99th percentile: 0.7346
100th percentile: 3.1334

License

This project is licensed under the MIT License. 


