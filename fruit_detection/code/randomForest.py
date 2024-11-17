from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar

# Load data
features_df = pd.read_csv("dataset/features_reduced.csv")
labels_df = pd.read_csv("dataset/processed_dataset_labels.csv")

# Features and labels
X = features_df
y = labels_df['state']  # Freshness label as the target variable

# Standardize data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Split data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3, random_state=42, stratify=y)

# Define model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Use 10-fold StratifiedKFold for cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# To store cross-validation scores and confusion matrix
cv_scores = []  
all_conf_matrix = np.zeros((2, 2))  # Assuming a binary classification problem

# Perform cross-validation with progress bar
for train_index, val_index in tqdm(kf.split(X_train, y_train), total=kf.get_n_splits(), desc="Cross-validation", ncols=100):
    X_kf_train, X_kf_val = X_train[train_index], X_train[val_index]
    y_kf_train, y_kf_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # Train model
    model.fit(X_kf_train, y_kf_train)
    y_pred = model.predict(X_kf_val)
    
    # Calculate accuracy and accumulate
    accuracy = accuracy_score(y_kf_val, y_pred)

    print(f"Accuracy of current fold: {accuracy:.4f}")

    cv_scores.append(accuracy)
    
    # Accumulate confusion matrix
    conf_matrix = confusion_matrix(y_kf_val, y_pred)
    all_conf_matrix += conf_matrix

# Calculate mean cross-validation score
cv_mean_score = np.mean(cv_scores)

# Output cross-validation results
print(f"RandomForest - Mean cross-validation score: {cv_mean_score:.4f}")
print(f"RandomForest - Cumulative confusion matrix:\n{all_conf_matrix}")

# Retrain the model using the full training set
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/best_random_forest_model.joblib")

# Load model and evaluate generalization ability on test set
loaded_model = joblib.load("model/best_random_forest_model.joblib")
y_test_pred = loaded_model.predict(X_test)

# Output evaluation results on test set
test_accuracy = accuracy_score(y_test, y_test_pred)
conf_matrix = confusion_matrix(y_test, y_test_pred)
class_report = classification_report(y_test, y_test_pred)

print("\nEvaluation results on test set:")
print(f"Test set accuracy: {test_accuracy:.4f}")
print(f"Test set confusion matrix:\n{conf_matrix}")
print(f"Test set classification report:\n{class_report}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Plot training set class distribution
plt.subplot(1, 2, 1)
y_train.value_counts().plot(kind='bar', color='skyblue', title="Training set class distribution")
plt.xlabel('Class')
plt.ylabel('Number of samples')

# Plot test set class distribution
plt.subplot(1, 2, 2)
y_test.value_counts().plot(kind='bar', color='lightgreen', title="Test set class distribution")
plt.xlabel('Class')
plt.ylabel('Number of samples')

plt.tight_layout()
plt.show()
