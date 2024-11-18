import os
import numpy as np
import cv2  
import tensorflow as tf
from tensorflow.keras import models
from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

# Load the trained model
model = models.load_model('cnn_model.h5')

# Define a function to load a specific image
def load_image(image_path, img_size=(128, 128)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)  
    img = cv2.resize(img, img_size) 
    img = img / 255.0  
    return np.expand_dims(img, axis=0)

# Choose an image 
folder_to_use = r'D:\Face_fake_detection\Data_files\full_ds\real'  
image_name = 'real_00001.jpg_face_1.jpg'  
image_path = os.path.join(folder_to_use, image_name)

# Define the original label of the image (for display purposes)
original_label = "real"  

# Load the image
selected_image = load_image(image_path)

# Define a prediction function for LIME
def predict_fn(images):
    return model.predict(images)

# Initialize LIME explainer
explainer = lime_image.LimeImageExplainer()

# Generate LIME explanation
explanation = explainer.explain_instance(
    selected_image[0],  
    predict_fn, 
    top_labels=4,  # Number of classes to explain
    hide_color=0, 
    num_samples=1000  # Number of samples to generate
)

# Plot the LIME explanation 
custom_labels = ["real", "easy", "mid", "hard"]
label_to_explain = 0  

# Display the explanation
temp, mask = explanation.get_image_and_mask(
    label=label_to_explain,  # Change this based on which label you want to explain
    positive_only=False,
    num_features=10,
    hide_rest=False
)

# Plot the original image label and LIME explanation
plt.figure(figsize=(10, 5))
# plt.suptitle(f"Original Image Label: {original_label}", fontsize=16)
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(selected_image[0])
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title(f"LIME Explanation - Label: {custom_labels[label_to_explain]}")
plt.imshow(mark_boundaries(temp, mask))
plt.axis('off')
plt.show()
