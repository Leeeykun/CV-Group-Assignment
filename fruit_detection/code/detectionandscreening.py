import cv2
import joblib
import numpy as np
from feature_extraction import extract_features
import os

# Resize image
def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size)

# Convert to HSV color space
def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Adjust brightness and contrast
def adjust_brightness_contrast(image, alpha=1.2, beta=30):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def preprocess_image(image):
    image = resize_image(image, size=(256, 256))
    image = convert_to_hsv(image)
    image = adjust_brightness_contrast(image)
    return image

# Load model, scaler and PCA
model = joblib.load("../model/best_random_forest_model.joblib")
scaler = joblib.load("../model/scaler.joblib")
pca = joblib.load("../model/pca_model.joblib")  

# Function to predict freshness
def predict_freshness(image, fruit_type):
    image = preprocess_image(image)
    features = extract_features(image, fruit_type)

    # Standardize using scaler
    features = scaler.transform([features])

    # Check features dimensions
    print(f"Feature dimensions after standardization: {features.shape}")

    # Ensure PCA input is 2-dimensional
    if len(features.shape) == 1:
        features = features.reshape(1, -1)  # Convert 1D data to 2D

    # Dimension reduction using PCA
    features = pca.transform(features)
    print(f"Feature dimensions after PCA: {features.shape}")

    prediction = model.predict(features)
    return prediction[0]

# Capture video from camera
def main():
    fruit_type = "banana"  # Replacement of fruit types detected by the camera
    cap = cv2.VideoCapture(0)  # Turn on the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predicting freshness
        predicted_state = predict_freshness(frame, fruit_type)
        if predicted_state is not None:
            print(f"Predicted freshness state: {predicted_state}")

        # Display the prediction results on the image
        cv2.putText(frame, f"Freshness: {predicted_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show image
        cv2.imshow('Fruit Freshness Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
