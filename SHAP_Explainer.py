import os
import numpy as np
import cv2  
import tensorflow as tf
from tensorflow.keras import models
import shap

# Load the trained model
model = models.load_model('cnn_model.h5')

# Define a function to load a specific image
def load_image(image_path, img_size=(128, 128)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)  
    if img is None:
        raise ValueError(f"Could not read the image: {image_path}")
    
    img = cv2.resize(img, img_size)  
    img = img / 255.0 
    return np.expand_dims(img, axis=0) 

# Choose an image 
folder_to_use = r'D:\Face_fake_detection\Data_files\full_ds\real'  
image_name = 'real_00001.jpg'  
image_path = os.path.join(folder_to_use, image_name)

# Load the image
selected_image = load_image(image_path)

# Prepare the SHAP model prediction function
def model_predict(x):
    return model.predict(x)

# Create a SHAP explainer
masker = shap.maskers.Image("inpaint_telea", selected_image[0].shape)
explainer = shap.Explainer(model_predict, masker)

# Calculate SHAP values 
shap_values = explainer(selected_image, max_evals=500, batch_size=50)

# Plot the SHAP values
custom_labels = ["real", "easy", "mid", "hard"]
shap.image_plot(shap_values, selected_image, labels=custom_labels)
