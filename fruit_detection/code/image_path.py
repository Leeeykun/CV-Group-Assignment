import os
import pandas as pd

# Define the dataset folder path
dataset_path = "dataset/original_dataset"

# Print the current directory
print("Current directory:", os.getcwd())

# Create a list to store file paths and labels
data = []

# Iterate through each folder
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    
    # Confirm it's a directory
    if os.path.isdir(folder_path):
        # Split the folder name to extract the state and fruit type
        parts = folder_name.split('_')
        if len(parts) == 2:
            state = parts[0]  # State label (e.g., fresh or stale)
            fruit_type = parts[1]  # Fruit type (e.g., apple, banana, orange)
            
            # Iterate through each image in the folder
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                # Add the path, state, and fruit type labels to the list
                data.append([file_path, state, fruit_type])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["image_path", "state", "fruit_type"])

# Save as a CSV file
df.to_csv("dataset/dataset_labels.csv", index=False)
print("Label data has been saved to dataset_labels.csv")
