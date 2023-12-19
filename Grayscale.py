import cv2
import numpy as np

image = None

# Preprocess the license plate image
def preprocess_license_plate(image_path):
    global image  # Use global to modify the global variable
    # Read the image
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    denoised_image = cv2.medianBlur(grayscale_image, 3)

    # Contrast enhancement
    equalized = cv2.equalizeHist(denoised_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(equalized)

    return enhanced

# Example usage
image_path = r"D:\Project\IOT\License_plate\motorbike_plate.png"
preprocessed_image = preprocess_license_plate(image_path)

# Display results (optional)
cv2.imshow("Preprocessed", preprocessed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
