import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re

def main():
    save_path = "/home/george/Documents/lava_loihi/plots/"
    csv_folder_path = "/media/george/T7 Shield/Neuromorphic Data/George/tests/simulator_tests/"  # Replace with your actual path
    csv_files = glob.glob(f"{csv_folder_path}*.csv")

    if f"{csv_folder_path}combined_frame.csv" not in csv_files:
        print("Combined csv not found. Combining frames...")
        data_frames = []

        for file in csv_files:
            df = pd.read_csv(file)
            # Extract 'Iteration' from the filename
            # Assuming the filename is in the format '{label}-iteration-{iteration_num}.csv'
            filename = os.path.basename(file)
            iteration_match = re.search(r'-iteration-(\d+)\.csv$', filename)
            if iteration_match:
                iteration_number = int(iteration_match.group(1))
            else:
                # Handle the case where the iteration number isn't found
                iteration_number = None
            df['Iteration'] = iteration_number

            # Since 'Target Label' is already in the data, no need to extract it from the filename
            data_frames.append(df)

        # Combine all data frames into one
        data = pd.concat(data_frames, ignore_index=True)

        # Convert data types
        data['Target Label'] = data['Target Label'].astype(str)
        data['Arm Speed'] = data['Arm Speed'].astype(str)
        data['Attempt'] = data['Attempt'].astype(int)
        data = data.fillna(0.0)

        data.to_csv(f"{csv_folder_path}/combined_frame.csv")
    else:
        print("Combined csv found. Loading...")
        data = pd.read_csv(f"{csv_folder_path}/combined_frame.csv")

    print("Data loaded")

    # Step 2: Plot frequency of 'Arm Speed' across the entire dataset
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Arm Speed', data=data, order=sorted(data['Arm Speed'].unique()), stat="proportion")
    plt.title('Frequency of Arm Speeds Across the Entire Dataset')
    plt.xlabel('Arm Speed')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_path}/arm_speeds.png", dpi=300)
    #plt.show()

    # Step 3: Plot frequency of 'Arm Speed' broken down by 'Target Label'
    plt.figure(figsize=(12, 6))
    sns.countplot(
        x="Arm Speed",
        hue="Target Label",
        data=data,
        order=sorted(data["Arm Speed"].unique()),
        stat="proportion",
    )
    plt.title('Frequency of Arm Speeds by Target Label')
    plt.xlabel('Arm Speed')
    plt.ylabel('Frequency')
    plt.legend(title='Target Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_path}/arm_speeds_label.png", dpi=300)
    #plt.show()

    # Step 4: Analyze how 'Entropy' changes over the 5 attempts across the entire dataset
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Attempt", y="Entropy", data=data, estimator="mean", errorbar="sd")
    plt.title('Entropy Across Attempts (1-5) in Entire Dataset')
    plt.xlabel('Attempt')
    plt.ylabel('Entropy')
    plt.xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    plt.savefig(f"{save_path}/entropy.png", dpi=300)
    #plt.show()

    # Step 5: Analyze how 'Entropy' changes over the 5 attempts for each 'Target Label'
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        x="Attempt",
        y="Entropy",
        hue="Target Label",
        data=data,
        estimator="mean",
        errorbar="sd",
    )
    plt.title('Entropy Across Attempts by Target Label')
    plt.xlabel('Attempt')
    plt.ylabel('Entropy')
    plt.legend(title='Target Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    plt.savefig(f"{save_path}/entropy_per_label.png", dpi=300)
    #plt.show()

    # Alternatively, use FacetGrid to create separate plots for each 'Target Label'
    g = sns.FacetGrid(data, col='Target Label', col_wrap=5, height=4)
    g.map_dataframe(sns.lineplot, x='Attempt', y='Entropy', estimator='mean')
    g.set_axis_labels('Attempt', 'Entropy')
    g.set_titles('Target Label: {col_name}')
    plt.tight_layout()
    #plt.show()

    # Step 6: Analyze how each of the 5 attempts affects other metrics (e.g., 'Confidence', 'Total Spikes')
    # Example for 'Confidence' across attempts
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        x="Attempt", y="Confidence", data=data, estimator="mean", errorbar="sd"
    )
    plt.title('Confidence Across Attempts (1-5) in Entire Dataset')
    plt.xlabel('Attempt')
    plt.ylabel('Confidence')
    plt.xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    plt.savefig(f"{save_path}/confidence.png", dpi=300)
    #plt.show()

    # Analyze 'Confidence' across attempts by 'Target Label'
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        x="Attempt",
        y="Confidence",
        hue="Target Label",
        data=data,
        estimator="mean",
        errorbar="sd",
    )
    plt.title('Confidence Across Attempts by Target Label')
    plt.xlabel('Attempt')
    plt.ylabel('Confidence')
    plt.legend(title='Target Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    plt.savefig(f"{save_path}/confidence_by_label.png", dpi=300)
    #plt.show()


if __name__ == "__main__":
    main()
