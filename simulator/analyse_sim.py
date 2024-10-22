import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Step 1: Read all CSV files
# Get a list of all CSV files matching the pattern
results_path = "/media/george/T7 Shield/Neuromorphic Data/George/tests/simulator_tests/"
csv_files = glob.glob(f"{results_path}/*-*-iteration-*.csv")

# Initialize an empty list to hold DataFrames
df_list = []

# Step 2: Read and combine all CSV files into a single DataFrame
for file in csv_files:
    df = pd.read_csv(file)
    # Extract mode, label, and test number from filename if needed
    filename = os.path.basename(file)
    mode, label, _, test_num_csv = filename.split('-')
    test_num = test_num_csv.split('.')[0]  # Remove '.csv' extension
    # Add extracted info to DataFrame if necessary
    df['Mode'] = mode
    df['Target Label'] = label
    df['Test Number'] = test_num
    df_list.append(df)

# Combine all DataFrames into one
data = pd.concat(df_list, ignore_index=True)

# Step 3: Ensure numeric columns are properly converted to numeric types for aggregation
numeric_columns = ['Arm Speed', 'Time Step', 'Confidence', 'Entropy', 'Total Spikes', 'Max Spikes', 'Attempt']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill NaN or null values with 0.0
data = data.fillna(0.0)

# Ensure 'Target Label' and 'Decision' are of the same datatype for comparison
data['Target Label'] = data['Target Label'].astype(str)
data['Decision'] = data['Decision'].astype(str)

# Normalize the 'Entropy' column
if data['Entropy'].max() != data['Entropy'].min():
    data['Entropy'] = (data['Entropy'] - data['Entropy'].min()) / (data['Entropy'].max() - data['Entropy'].min())

# Step 4: Add a 'Correct' column to indicate if the decision was correct
# 'Correct' is 1 if 'Target Label' is equal to 'Decision', otherwise 0
data['Correct'] = (data['Target Label'] == data['Decision']).astype(int)

# Step 5: Create a separate DataFrame for accuracy over time step for each mode and attempt
accuracy_data = data.groupby(['Mode', 'Time Step', 'Attempt'], as_index=False).agg({'Correct': 'mean'}).rename(columns={'Correct': 'Accuracy'})

# Step 6: Drop non-numeric columns that are not needed for aggregation
non_numeric_columns = ['Filename', 'Test Number', 'Target Label', 'Decision']
data = data.drop(columns=[col for col in non_numeric_columns if col in data.columns])

# Step 7: Group the data by 'Mode', 'Time Step', and 'Attempt' for averaging across all target labels and arm speeds
# Use only numeric columns for averaging
averaged_data = data.groupby(['Mode', 'Time Step', 'Attempt'], as_index=False).mean()

# Print out the averaged data used for plotting
print("Averaged Data:")
print(averaged_data)

# Step 8: Plot Averaged Confidence and Entropy Over Time Step for Comparison
if not averaged_data.empty:
    modes = averaged_data['Mode'].unique()
    for mode in modes:
        mode_data = averaged_data[averaged_data['Mode'] == mode]
        
        # Plot Averaged Confidence for Attempts 1 and 2 on the same graph
        plt.figure(figsize=(12, 8))
        attempt_1_data = mode_data[mode_data['Attempt'] == 1]
        attempt_2_data = mode_data[mode_data['Attempt'] == 2]
        
        # Plot Attempt 1 and Attempt 2 Confidence
        if not attempt_1_data.empty:
            plt.plot(attempt_1_data['Time Step'], attempt_1_data['Confidence'], label='Attempt 1', linestyle='--')
        if not attempt_2_data.empty:
            plt.plot(attempt_2_data['Time Step'], attempt_2_data['Confidence'], label='Attempt 2', linestyle='-')
        
        plt.title(f'Averaged Confidence vs. Time Step for Mode: {mode}')
        plt.xlabel('Time Step')
        plt.ylabel('Averaged Confidence')
        plt.legend()
        #plt.show()
        
        # Plot Averaged Entropy for Attempts 1 and 2 on the same graph
        plt.figure(figsize=(12, 8))
        if not attempt_1_data.empty:
            plt.plot(attempt_1_data['Time Step'], attempt_1_data['Entropy'], label='Attempt 1', linestyle='--')
        if not attempt_2_data.empty:
            plt.plot(attempt_2_data['Time Step'], attempt_2_data['Entropy'], label='Attempt 2', linestyle='-')
        
        plt.title(f'Averaged Entropy vs. Time Step for Mode: {mode}')
        plt.xlabel('Time Step')
        plt.ylabel('Averaged Entropy')
        plt.legend()
        #plt.show()

# Step 9: Plot Averaged Entropy Across All Modes for Easy Comparison (Attempt 2 Only)
if not averaged_data.empty:
    plt.figure(figsize=(14, 10))
    for mode in modes:
        mode_data = averaged_data[averaged_data['Mode'] == mode]
        attempt_2_data = mode_data[mode_data['Attempt'] == 2]
        
        # Plot Attempt 2 Entropy for each mode
        if not attempt_2_data.empty:
            plt.plot(attempt_2_data['Time Step'], attempt_2_data['Entropy'], label=f'Mode: {mode} - Attempt 2', linestyle='-')

    plt.title('Averaged Entropy vs. Time Step Across All Modes (Attempt 2 Only)')
    plt.xlabel('Time Step')
    plt.ylabel('Averaged Entropy')
    plt.legend()
    #plt.show()

# Step 10: Plot Averaged Confidence Across All Modes for Easy Comparison (Attempt 2 Only)
if not averaged_data.empty:
    plt.figure(figsize=(14, 10))
    for mode in modes:
        mode_data = averaged_data[averaged_data['Mode'] == mode]
        attempt_2_data = mode_data[mode_data['Attempt'] == 2]
        
        # Plot Attempt 2 Confidence for each mode
        if not attempt_2_data.empty:
            plt.plot(attempt_2_data['Time Step'], attempt_2_data['Confidence'], label=f'Mode: {mode} - Attempt 2', linestyle='-')

    plt.title('Averaged Confidence vs. Time Step Across All Modes (Attempt 2 Only)')
    plt.xlabel('Time Step')
    plt.ylabel('Averaged Confidence')
    plt.legend()
    #plt.show()

# Step 11: Plot Average Difference in Entropy Between Attempts 1 and 2 for Each Mode as a Bar Graph
if not averaged_data.empty:
    diff_entropy_list = []
    for mode in modes:
        mode_data = averaged_data[averaged_data['Mode'] == mode]
        attempt_1_data = mode_data[mode_data['Attempt'] == 1]
        attempt_2_data = mode_data[mode_data['Attempt'] == 2]
        
        if not attempt_1_data.empty and not attempt_2_data.empty:
            # Calculate the average difference between Attempt 2 and Attempt 1 for Entropy
            avg_diff_entropy = (attempt_2_data['Entropy'].values - attempt_1_data['Entropy'].values).mean()
            diff_entropy_list.append((mode, avg_diff_entropy))

    # Convert the list to a DataFrame for easier plotting
    diff_entropy_df = pd.DataFrame(diff_entropy_list, columns=['Mode', 'Average Entropy Difference'])

    # Plot the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(diff_entropy_df['Mode'], diff_entropy_df['Average Entropy Difference'], color='skyblue')
    plt.title('Average Difference in Entropy Between Attempts 1 and 2 for Each Mode')
    plt.xlabel('Mode')
    plt.ylabel('Average Entropy Difference (Attempt 2 - Attempt 1)')
    plt.xticks(rotation=45)
    #plt.show()

# Step 12: Plot Average Accuracy Over Time Step for Each Mode and Attempt
if not accuracy_data.empty:
    plt.figure(figsize=(14, 10))
    for mode in accuracy_data['Mode'].unique():
        mode_data = accuracy_data[accuracy_data['Mode'] == mode]
        for attempt in mode_data['Attempt'].unique():
            attempt_data = mode_data[mode_data['Attempt'] == attempt]
            plt.plot(attempt_data['Time Step'], attempt_data['Accuracy'], label=f'Mode: {mode} - Attempt: {attempt}', linestyle='-')

    plt.title('Average Accuracy vs. Time Step for Different Modes and Attempts')
    plt.xlabel('Time Step')
    plt.ylabel('Average Accuracy')
    plt.legend()
    #plt.show()

# Step 13: Find Tc and Th for Each Mode and Attempt, and Output as a LaTeX Table
latex_table_data = []
if not accuracy_data.empty:
    for mode in accuracy_data['Mode'].unique():
        mode_data = accuracy_data[accuracy_data['Mode'] == mode]
        for attempt in mode_data['Attempt'].unique():
            attempt_data = mode_data[mode_data['Attempt'] == attempt]
            
            # Find the time step where accuracy reaches its maximum value (Tc)
            tc = attempt_data.loc[attempt_data['Accuracy'].idxmax(), 'Time Step']
            max_accuracy = attempt_data['Accuracy'].max()
            
            # Find the time step where accuracy first exceeds 65% (Th)
            th_data = attempt_data[attempt_data['Accuracy'] > 0.65]
            if not th_data.empty:
                th = th_data.iloc[0]['Time Step']
            else:
                th = 'Not reached'
            
            # Append data for LaTeX table
            latex_table_data.append([mode, attempt, tc, th, max_accuracy])

# Create LaTeX table from the collected data
latex_df = pd.DataFrame(latex_table_data, columns=['Mode', 'Attempt', 'Tc', 'Th', 'Max Accuracy'])
print("\nLaTeX Table:\n")
print(latex_df.to_latex(index=False))