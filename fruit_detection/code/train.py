from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load data
features_df = pd.read_csv("dataset/features_reduced.csv")
labels_df = pd.read_csv("dataset/processed_dataset_labels.csv")

# Features and labels
X = features_df
y = labels_df['state']  # Freshness label as the target variable

# Standardize data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

# Use 10-fold StratifiedKFold for cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Store results for each model
results = {}

for model_name, model in models.items():
    cv_scores = []  # Store scores for each cross-validation
    all_conf_matrix = np.zeros((2, 2))  # Store confusion matrices for all folds (assuming binary classification)
    all_class_report = ""  # Store classification report
    
    # Perform cross-validation
    for train_index, val_index in kf.split(X_standardized, y):
        X_train, X_val = X_standardized[train_index], X_standardized[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_val, y_pred)
        cv_scores.append(accuracy)
        
        # Accumulate confusion matrix
        conf_matrix = confusion_matrix(y_val, y_pred)
        all_conf_matrix += conf_matrix
        
        # Classification report
        class_report = classification_report(y_val, y_pred)
        all_class_report += class_report
    
    # Store average cross-validation score for the model
    results[model_name] = {
        "cv_mean_score": np.mean(cv_scores),
        "confusion_matrix": all_conf_matrix,
        "classification_report": all_class_report
    }
    
    # Output average cross-validation score
    print(f"{model_name} - Average cross-validation score: {results[model_name]['cv_mean_score']:.4f}")
    
# Print final evaluation results for each model (confusion matrix and classification report)
print("\nModel Evaluation Results:")
for model_name, result in results.items():
    print(f"\n{model_name} - Confusion Matrix:\n{result['confusion_matrix']}")
    print(f"{model_name} - Classification Report:\n{result['classification_report']}")

# Find the best performing model
best_model_name = max(results, key=lambda x: results[x]["cv_mean_score"])
print(f"\nThe best model is: {best_model_name}, with an average cross-validation score of: {results[best_model_name]['cv_mean_score']:.4f}")
