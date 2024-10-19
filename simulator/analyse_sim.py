#!/bin/bash
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
import numpy as np

# Step 1: Read and combine all CSV files
csv_folder_path = "/media/george/T7 Shield/Neuromorphic Data/George/tests/simulator_tests/"  # Replace with your actual path
csv_files = glob.glob(os.path.join(csv_folder_path, '*.csv'))

data_frames = []
num_tests = len(csv_files)
time_steps = None

# Step 2: Initialize accuracy array
accuracy_array = []

for file in csv_files:
    df = pd.read_csv(file)
    # Extract mode and iteration from the filename
    filename = os.path.basename(file)
    match = re.match(r'(?P<mode>.*?)-(?P<sim_label>.*?)-iteration-(?P<test_num>\d+)\.csv$', filename)
    if match:
        df['Mode'] = match.group('mode')
        df['Iteration'] = int(match.group('test_num'))
    # Add a time step column using the index
    df['Time Step'] = df.index
    data_frames.append(df)

    # Initialize time_steps if not set
    if time_steps is None:
        time_steps = len(df)

    # Calculate accuracy for each time step in the current file
    accuracy = (df['Target Label'] == df['Decision']).astype(int).to_numpy()
    accuracy_array.append(accuracy)

# Convert accuracy array to numpy array
accuracy_array = np.array(accuracy_array)  # Shape: (num_tests, time_steps)

# Calculate average accuracy across all tests for each time step
average_accuracy_over_time = np.mean(accuracy_array, axis=0)

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

# Step 3: Separate analysis for each mode
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
    sns.lineplot(x='Attempt', y='Entropy', data=mode_data, estimator='mean', errorbar='sd')
    plt.title(f'Entropy Across Attempts (1-5) for Mode: {mode}')
    plt.xlabel('Attempt')
    plt.ylabel('Entropy')
    plt.xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    plt.savefig(f'entropy_attempts_mode_{mode}.png')
    plt.close()

    # Plot average accuracy over time steps for each mode
    plt.figure(figsize=(12, 6))
    plt.plot(range(time_steps), average_accuracy_over_time, label=f'Mode: {mode}', marker='o')
    plt.title(f'Average Accuracy Over Time Steps for Mode: {mode}')
    plt.xlabel('Time Step')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(f'average_accuracy_over_time_mode_{mode}.png')
    plt.close()

    # Plot average metrics over time steps for each mode
    metrics_to_plot = ['Entropy', 'Confidence', 'Total Spikes']
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        avg_metric_over_time = mode_data.groupby('Time Step')[metric].mean()
        plt.plot(avg_metric_over_time.index, avg_metric_over_time.values, label=f'Mode: {mode}', marker='o')
        plt.title(f'Average {metric} Over Time Steps for Mode: {mode}')
        plt.xlabel('Time Step')
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(f'average_{metric.lower().replace(" ", "_")}_over_time_mode_{mode}.png')
        plt.close()

# Step 4: Normalise entropy using global min and max for comparison between modes
entropy_min = data['Entropy'].min()
entropy_max = data['Entropy'].max()
data['Normalized Entropy'] = (data['Entropy'] - entropy_min) / (entropy_max - entropy_min)

# Step 5: Comparison between modes for metrics
# Plot entropy for each target label, for each mode
for mode in modes:
    mode_data = data[data['Mode'] == mode]
    target_labels = mode_data['Target Label'].unique()
    for label in target_labels:
        label_data = mode_data[mode_data['Target Label'] == label]
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Attempt', y='Entropy', data=label_data, estimator='mean', errorbar='sd')
        plt.title(f'Entropy Across Attempts for Mode: {mode}, Target Label: {label}')
        plt.xlabel('Attempt')
        plt.ylabel('Entropy')
        plt.xticks([1, 2, 3, 4, 5])
        plt.tight_layout()
        plt.savefig(f'entropy_attempts_mode_{mode}_target_label_{label}.png')
        plt.close()

metrics = ['Normalized Entropy', 'Confidence', 'Total Spikes']
for metric in metrics:
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Attempt', y=metric, hue='Mode', data=data, estimator='mean', errorbar='sd')
    plt.title(f'Comparison of {metric.replace("Normalized ", "")} Across Attempts by Mode')
    plt.xlabel('Attempt')
    plt.ylabel(metric)
    plt.legend(title='Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    plt.savefig(f'comparison_{metric.lower().replace(" ", "_")}_by_mode.png')
    plt.close()

# Step 5: Save averaged metrics to CSV
averaged_metrics = data.groupby(['Mode', 'Target Label', 'Attempt']).agg(
    {'Entropy': 'mean', 'Confidence': 'mean', 'Total Spikes': 'mean'}
).reset_index()

averaged_metrics.to_csv('averaged_metrics_by_mode.csv', index=False)
