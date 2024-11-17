import cv2
import numpy as np

# Color feature extraction function
def extract_color_features(image):
    """
    Calculate the color histogram features of an image.
    Normalize the HSV histogram of the image and flatten it into a one-dimensional feature vector.
    """
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # Normalize and flatten into a one-dimensional array
    return hist

# Shape feature extraction function
def extract_shape_features(image, fruit_type):
    """
    Extract suitable shape features based on the type of fruit.
    For bananas, extract aspect ratio and orientation; for other fruits, extract area, perimeter, and roundness.
    """
    # Default feature values
    default_shape_features = [0, 0, 0]  # Aspect ratio, orientation, roundness

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        contour = max(contours, key=cv2.contourArea)  # Select the largest contour
        
        if fruit_type == "banana":
            # Extract aspect ratio and orientation angle for bananas
            aspect_ratio = calculate_aspect_ratio(contour)
            orientation = calculate_orientation(contour)
            return [aspect_ratio, orientation, 0]
        
        else:
            # Extract area, perimeter, and roundness for other fruits
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            roundness = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter != 0 else 0
            return [area, perimeter, roundness]
    else:
        # Return default values if no contour is found
        return [0, 0, 0]
    
# Helper function: calculate aspect ratio
def calculate_aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return float(w) / h if h != 0 else 0

# Helper function: calculate orientation angle
def calculate_orientation(contour):
    rect = cv2.minAreaRect(contour)
    return rect[2]

# Comprehensive feature extraction function
def extract_features(image, fruit_type):
    """
    Extract color and shape features of an image and combine them into a feature vector.
    """
    color_features = extract_color_features(image)
    shape_features = extract_shape_features(image, fruit_type)
    return np.concatenate([color_features, shape_features])  # Combine color and shape features into a single feature vector
