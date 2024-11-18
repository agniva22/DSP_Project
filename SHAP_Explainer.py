import os
import numpy as np
import cv2  # OpenCV for image loading
import tensorflow as tf
from tensorflow.keras import models
import shap

# Load the trained model
# model = models.load_model('my_cnn_model.h5')
model = models.load_model('NEW_cnn_model.h5')


# Define a function to load a specific image
def load_image(image_path, img_size=(128, 128)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)  # Read image
    if img is None:
        raise ValueError(f"Could not read the image: {image_path}")
    
    img = cv2.resize(img, img_size)  # Resize image
    img = img / 255.0  # Normalize the image
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Choose an image from one of the folders
folder_to_use = r'D:\Face_fake_detection\Data_files\full_ds\real'  # Example: Use a real image
image_name = 'real_00001.jpg'  # Replace with the actual image name
image_path = os.path.join(folder_to_use, image_name)

# Print the image path for debugging
print(f"Loading image from: {image_path}")

# Load the image
selected_image = load_image(image_path)

# Prepare the SHAP model prediction function
def model_predict(x):
    return model.predict(x)

# Create a SHAP explainer
masker = shap.maskers.Image("inpaint_telea", selected_image[0].shape)
explainer = shap.Explainer(model_predict, masker)

# Calculate SHAP values for the selected image
shap_values = explainer(selected_image, max_evals=500, batch_size=50)

# Plot the SHAP values
custom_labels = ["real", "easy", "mid", "hard"]
shap.image_plot(shap_values, selected_image, labels=custom_labels)
