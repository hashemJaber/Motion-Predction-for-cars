'''!pip install omegaconf
!pip install  pyntcloud
!pip install open3d
!pip install OpenCV
!pip install Plotly
!pip install psutil requests
!apt install aria2
!apt-get install -y orca
'''
### I assume the above will already be set up in your enviroment
import tarfile
import os
import pyarrow.feather as feather
import numpy as np
import plotly.graph_objs as go
from ipywidgets import Button, VBox
from IPython.display import display, clear_output
from PIL import Image
import matplotlib.pyplot as pyplot
import pyarrow.feather as feather
import open3d as o3d
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader

list_of_parquet_files = os.listdir("/argotrain/train")
#"/argotrain/train/0000b0f9-99f9-4a1f-a231-5be9e4c523f7/scenario_0000b0f9-99f9-4a1f-a231-5be9e4c523f7.parquet"
list_of_parquet_files_string=[]
for i in range(199908):
  list_of_parquet_files_string.append("/argotrain/train/"+list_of_parquet_files[i]+"/scenario_"+list_of_parquet_files[i]+".parquet")



# Parameters
covered_time = 80      # Observed timesteps
future_time = 30       # Future timesteps
max_objects = 80       # Maximum number of objects per scenario, made through analysis on average count of objects per scneriar, mean was 55.5
bins = 120              # Number of bins for discretization, made arbitrarly, no reason behind it
batch_size = 16        # Batch size for training
num_epochs = 100        # Number of training epochs
learning_rate = 1e-4   # Learning rate for optimizer

# Pre-determined Global Bin Ranges (Set based on our prior analysis, check the ReadMe for more info at https://github.com/hashemJaber/Motion-Predction-for-cars)
x_range_min = 0.0
x_range_max = 41.7139
y_range_min = -5.5628
y_range_max = 6.2710
heading_range_min = -0.7035
heading_range_max = 0.7346

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# 1. Extract World Trajectories
def extract_world_trajectories(parquet_path:str, max_objects:int=80, observed_timesteps:int=80):
    """
    Extract world trajectories for the observed timesteps.
    Args:
        parquet_path: str  (Path to the .parquet file)
        max_objects: int (Maximum number of objects to extract)
        observed_timesteps: int (The number of timesteps to include as input (i.e., the history up to t))
    Returns:
        World trajectory array of shape (max_objects, observed_timesteps, feature_dim).
    """
    df = pd.read_parquet(parquet_path)
    df = df.sort_values(by=['track_id', 'timestep'])
    features = ['position_x', 'position_y', 'heading', 'velocity_x', 'velocity_y', 'object_category']

    grouped = df.groupby('track_id')
    objects = [object_group[features].to_numpy()[:observed_timesteps]
               for _, object_group in grouped if len(object_group) >= observed_timesteps]


    # Pad or truncate to max_objects, maybe a bad opition subject to change
    num_objects = min(len(objects), max_objects)
    objects_array = np.zeros((max_objects, observed_timesteps, len(features)))
    objects_array[:num_objects] = objects[:num_objects]
    #print("objects_array: ",objects_array.shape)
    #print("extract_world_trajectories: ",objects_array)

    return objects_array



# 2. Extract Focused Object Trajectory
def extract_focused_object_trajectory(parquet_path:str, observed_timesteps:int=80):
    """
    Extract the focused object's trajectory up to the observed timesteps.

    NOTE: THIS MIGHT SEEM REDUNTANT FOR NOW, DUE TO THE WORLD TRAJECTORIES ALREADY INCLUDING THIS INFO
    BUT THIS IS SUBJECT TO CHANGE ONCE WE RESOLVE THE HD MAP REPRESENATION

    Args:
        parquet_path: str (Path to the .parquet file)
        observed_timesteps: int (The number of timesteps to include as input (i.e., the history up to t))

    Returns:
        Focused object trajectory array of shape (observed_timesteps, feature_dim).
    """
    df = pd.read_parquet(parquet_path)
    focal_track_id = df['focal_track_id'].iloc[0]
    focused_object_df = df[df['track_id'] == focal_track_id]

    features = ['position_x', 'position_y', 'heading', 'velocity_x', 'velocity_y', 'object_category']
    focused_object_df=focused_object_df[features].to_numpy()[:observed_timesteps]
    #print("focused_object_df: ",focused_object_df.shape)

    # Pad or truncate to max_objects
    #num_objects = min(len(objects), max_objects
    #print("focused_object_df: ",focused_object_df.shape)
    #print("focused_object_df: ",focused_object_df[:1])
    return focused_object_df