import cv2
import pandas as pd
import numpy as np
import joblib  # For saving models
from tqdm import tqdm  # Used to display a progress bar
from feature_extraction import extract_features
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Loading dataset labels
data = pd.read_csv("dataset/processed_dataset_labels.csv")  # Using preprocessed label files

# Define a list of saved features
features_list = []
labels_list = []

# Iterate through the dataset and display the progress bar
print("Of the extracted features...")
for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing images"):
    image_path = row['processed_image_path']  # Paths using processed images
    state = row['state']  # Freshness Label
    fruit_type = row['fruit_type']  # Fruit type

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Unable to read image: {image_path}")
        continue  # Skip this image and its labels

    # Extraction characteristics
    features = extract_features(image, fruit_type)

    # If the feature is invalid (e.g. it is NaN or all zeros), the sample is skipped
    if np.any(np.isnan(features)) or np.all(features == 0):
        print(f"Feature extraction failed, skip image: {image_path}")
        continue

    # Preservation of valid features and labelling
    features_list.append(features)
    labels_list.append([state, fruit_type])

# Ensure that the number of features and labels are consistent
if len(features_list) != len(labels_list):
    print("WARNING: Inconsistent number of features and labels!")
else:
    print("The number of features and labels is consistent.")

# Convert to DataFrame
features_df = pd.DataFrame(features_list)
labels_df = pd.DataFrame(labels_list, columns=["state", "fruit_type"])

# Deletes rows containing NaN values and removes labels simultaneously
print(f"Shape of feature data before NaN removal: {features_df.shape}, Shape of label data: {labels_df.shape}")
features_df = features_df.dropna()

# Simultaneous deletion of rows in labels_df using indexes in features_df
labels_df = labels_df.loc[features_df.index]

# Check if there are still NaN values
if np.any(np.isnan(features_df)):
    print("Warning: There are still NaN values in the feature matrix!")
else:
    print("There are no NaN values in the feature matrix.")

# Standardize the data
print("Standardizing feature data...")
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features_df)

# Save the standardization model
joblib.dump(scaler, "model/scaler.joblib")
print("StandardScaler has been saved as scaler.joblib")

# Perform PCA dimensionality reduction
print("Performing PCA dimensionality reduction...")
pca = PCA(n_components=0.95)  # Retain 95% of the variance
features_reduced = pca.fit_transform(features_standardized)

# Save the PCA model
joblib.dump(pca, "model/pca_model.joblib")
print("PCA model has been saved as pca_model.joblib")

# Ensure that the reduced data is two-dimensional
print(f"Shape of reduced features: {features_reduced.shape}")

# Save the reduced features as a DataFrame and save to a CSV file
print("Saving reduced features and labels...")
features_reduced_df = pd.DataFrame(features_reduced)
features_reduced_df.to_csv("dataset/features_reduced.csv", index=False)
labels_df.to_csv("dataset/labels.csv", index=False)

print("Reduced features and labels have been saved to features_reduced.csv and labels.csv")
