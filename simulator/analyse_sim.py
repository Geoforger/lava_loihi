#!/bin/bash
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
import numpy as np

# Step 1: Read and combine all CSV files
csv_folder_path = 'path/to/your/csv_folder'  # Replace with your actual path
csv_files = glob.glob(os.path.join(csv_folder_path, '*.csv'))

data_frames = []

for file in csv_files:
    df = pd.read_csv(file)
    # Extract mode and iteration from the filename
    filename = os.path.basename(file)
    match = re.match(r'(?P<mode>.*?)-(?P<sim_label>.*?)-iteration-(?P<test_num>\d+)\.csv$', filename)
    if match:
        df['Mode'] = match.group('mode')
        df['Iteration'] = int(match.group('test_num'))
    data_frames.append(df)

# Combine all data frames into one
data = pd.concat(data_frames, ignore_index=True)

# Replace infinite values with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Replace all NaN or empty values with 0.0
data.fillna(0.0, inplace=True)

# Convert data types
data['Target Label'] = data['Target Label'].astype(str)
data['Arm Speed'] = data['Arm Speed'].astype(str)
data['Attempt'] = data['Attempt'].astype(int)

# Step 2: Separate analysis for each mode
modes = data['Mode'].unique()
for mode in modes:
    mode_data = data[data['Mode'] == mode]

    # Plot frequency of 'Arm Speed' for each mode
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Arm Speed', data=mode_data, order=sorted(mode_data['Arm Speed'].unique()))
    plt.title(f'Frequency of Arm Speeds for Mode: {mode}')
    plt.xlabel('Arm Speed')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'frequency_arm_speed_mode_{mode}.png')
    plt.close()

    # Plot entropy over attempts for each mode
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Attempt', y='Entropy', data=mode_data, estimator='mean', ci='sd')
    plt.title(f'Entropy Across Attempts (1-5) for Mode: {mode}')
    plt.xlabel('Attempt')
    plt.ylabel('Entropy')
    plt.xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    plt.savefig(f'entropy_attempts_mode_{mode}.png')
    plt.close()

# Step 3: Comparison between modes for metrics
metrics = ['Entropy', 'Confidence', 'Total Spikes']
for metric in metrics:
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Attempt', y=metric, hue='Mode', data=data, estimator='mean', ci='sd')
    plt.title(f'Comparison of {metric} Across Attempts by Mode')
    plt.xlabel('Attempt')
    plt.ylabel(metric)
    plt.legend(title='Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    plt.savefig(f'comparison_{metric}_by_mode.png')
    plt.close()

# Step 4: Save averaged metrics to CSV
averaged_metrics = data.groupby(['Mode', 'Target Label', 'Attempt']).agg(
    {'Entropy': 'mean', 'Confidence': 'mean', 'Total Spikes': 'mean'}
).reset_index()

averaged_metrics.to_csv('averaged_metrics_by_mode.csv', index=False)