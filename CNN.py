import os
import numpy as np
import cv2  
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


folders = [
    r'./Data_files/full_ds/real',
    r'./Data_files/Augmented_ds/easy',
    r'./Data_files/Augmented_ds/mid',
    r'./Data_files/Augmented_ds/hard'
]
labels = ['real', 'easy', 'mid', 'hard']

# Load images and labels
def load_data(folders, labels, img_size=(128, 128)):
    images = []
    image_labels = []

    for folder, label in zip(folders, labels):
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            if img_path.endswith('.jpg') or img_path.endswith('.png'):
                img = cv2.imread(img_path)  
                img = cv2.resize(img, img_size)
                images.append(img)
                image_labels.append(label)

    return np.array(images), np.array(image_labels)


X, y = load_data(folders, labels)
X = X / 255.0  

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Proposed CNN Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(labels), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.2f}')

# Save the model
model.save('cnn_model.h5')
