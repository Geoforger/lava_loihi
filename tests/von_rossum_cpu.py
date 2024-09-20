import numpy as np
import os
import glob
from numba import njit
import math

# Define Van Rossum distance parameters (you can tweak these based on your data)
tau = 20.0  # decay constant for the Van Rossum kernel

# Step 1: Load the dataset
def load_dataset(folder_path):
    file_paths = glob.glob(os.path.join(folder_path, "*_on.pickle.npy"))
    print(f"Number of samples found: {len(file_paths)}")
    dataset = []
    
    for file_path in file_paths:
        # Extract force, speed, texture, and trial from filename
        file_name = os.path.basename(file_path)
        parts = file_name.split('-')
        speed = int(parts[-3])
        texture = int(parts[-2])
        
        data = np.load(file_path, allow_pickle=True)
        dataset.append((speed, texture, data))
    
    return dataset

# Step 2: Concatenate spike trains across the 2D array
def concatenate_spike_trains(sample):
    concatenated_spike_train = []
    
    # Loop through each cell in the 2D array and concatenate the spike trains
    for x in range(sample.shape[0]):
        for y in range(sample.shape[1]):
            concatenated_spike_train.extend(sample[x, y])
    
    return np.array(concatenated_spike_train, dtype=np.float64)

@njit
def convolve_spike_train(spike_train, tau):
    result = np.zeros_like(spike_train, dtype=np.float64)
    for i, t in enumerate(spike_train):
        result[i] = np.exp(-t / tau)
    return result

@njit
def compute_van_rossum_distances(convolved_tex1, convolved_tex2, tau):
    rows = convolved_tex1.shape[0]
    cols = convolved_tex2.shape[0]
    result_matrix = np.zeros((rows, cols), dtype=np.float64)
    
    for i in range(rows):
        for j in range(cols):
            dist = 0.0
            for k in range(convolved_tex1.shape[1]):
                diff = convolved_tex1[i, k] - convolved_tex2[j, k]
                dist += diff * diff
            result_matrix[i, j] = math.sqrt(dist)
    
    return result_matrix

def cpu_compute_all_distances(tex1_trials, tex2_trials, tau):
    # Step 1: Convolve spike trains and get their maximum length
    convolved_samples_tex1 = [convolve_spike_train(concatenate_spike_trains(trial), tau) for trial in tex1_trials]
    convolved_samples_tex2 = [convolve_spike_train(concatenate_spike_trains(trial), tau) for trial in tex2_trials]

    # Determine the maximum length across all convolved samples
    max_len = max(max([len(sample) for sample in convolved_samples_tex1]), max([len(sample) for sample in convolved_samples_tex2]))

    # Step 2: Pad or truncate all samples to ensure uniform length
    padded_tex1 = [np.pad(sample, (0, max_len - len(sample)), 'constant') if len(sample) < max_len else sample[:max_len] for sample in convolved_samples_tex1]
    padded_tex2 = [np.pad(sample, (0, max_len - len(sample)), 'constant') if len(sample) < max_len else sample[:max_len] for sample in convolved_samples_tex2]

    convolved_tex1 = np.array(padded_tex1)
    convolved_tex2 = np.array(padded_tex2)

    result_matrix = compute_van_rossum_distances(convolved_tex1, convolved_tex2, tau)
    
    return result_matrix

# Step 4: Compare all textures across trials and speeds
def compare_textures(dataset, tau):
    textures = [
        "Mesh", "Felt", "Cotton", "Nylon", "Fur",
        "Wood", "Acrylic", "FashionFabric", "Wool", "Canvas"
    ]
    texture_count = 10
    speed_values = [15, 25, 35, 45, 55]

    # Initialize output matrix
    output_matrix = np.zeros((texture_count, texture_count, len(speed_values)))

    # Loop through speeds
    for i, speed in enumerate(speed_values):
        print(f"Comparing Textures at speed: {speed}")
        for tex1 in range(texture_count):
            for tex2 in range(tex1, texture_count):
                print(f"Comparing textures: {textures[tex1]} and {textures[tex2]}")
                
                # Get all trials for tex1 and tex2 at the current speed
                trials_tex1 = [data[-1] for data in dataset if data[1] == tex1 and data[0] == speed]
                trials_tex2 = [data[-1] for data in dataset if data[1] == tex2 and data[0] == speed]
                
                # Compute distances using CPU
                result_matrix = cpu_compute_all_distances(trials_tex1, trials_tex2, tau)

                # Average the distances for the current texture pair and speed
                avg_distance = np.mean(result_matrix)
                output_matrix[tex1, tex2, i] = avg_distance
                output_matrix[tex2, tex1, i] = avg_distance
                print(f"Pair average distance: {avg_distance}")

    return output_matrix

def main():
    DATASET_PATH = "/media/george/T7 Shield/Neuromorphic Data/George/preprocessed_new_dataset/"
    OUTPUT_PATH = "/media/george/T7 Shield/Neuromorphic Data/George/tests/dataset_analysis/"
    
    # Example usage
    print("Loading dataset...")
    dataset = load_dataset(DATASET_PATH)
    print("Loaded dataset")
    print("Performing comparisons...")
    result_matrix = compare_textures(dataset, tau)
    print("Finished comparisons")

    # Save the result as a .npy file
    np.save(f"{OUTPUT_PATH}/tex_tex_speed_similarity_data.npy", result_matrix)

if __name__ == "__main__":
    main()
