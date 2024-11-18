# Fruit Freshness Detection Project

This project focuses on detecting the freshness of fruits using machine learning techniques. The dataset consists of images of different types of fruits in various states of freshness. The project involves preprocessing the images, extracting features, training different machine learning models, and evaluating their performance.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Image Preprocessing](#image-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Model Deployment](#model-deployment)
- [Usage](#usage)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fruit-freshness-detection.git
   cd fruit-freshness-detection
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The dataset consists of images of fruits categorized by type and freshness state. Before running the project, ensure the dataset is organized as follows:

```
dataset/
├── original_dataset/
│   ├── fresh_apple/
│   ├── stale_apple/
│   ├── fresh_banana/
│   └── stale_banana/
│   └── ...
└── processed_dataset/
    └── (processed images will be saved here)
```

## Image Preprocessing

1. Run the script to generate labels for the original dataset:
   ```bash
   python generate_labels.py
   ```

2. Run the image preprocessing script to resize, convert color spaces, adjust brightness and contrast, and save the processed images:
   ```bash
   python image_preprocess.py
   ```

## Feature Extraction

Run the feature extraction script to extract color and shape features from the processed images:
```bash
python feature_extraction.py
```

## Model Training and Evaluation

1. Train and evaluate multiple models (Random Forest, SVM, KNN, Logistic Regression) using  cross-validation:
   ```bash
   python train.py
   ```

2. Train the best model (Random Forest) on the full training set and evaluate it on the test set:
   ```bash
   python randomForest.py
   ```

## Model Deployment

1. Save the trained model, scaler, and PCA for deployment:
   ```bash
   python feature_extraction_and_pca.py
   ```

2. Use the saved model to predict the freshness of fruits in real-time using a webcam:
   ```bash
   python detectionandscreening.py
   ```

## Usage

1. Run the script to start the webcam and predict the freshness of fruits in real-time:
   ```bash
   python detectionandscreening.py
   ```

2. Press 'q' to exit the webcam view.


