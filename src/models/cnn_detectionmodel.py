import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the CNN architecture
def create_model():
    model = models.Sequential()

     # Convolutional Layer 1: 3x3 Kernel, 32 Filters
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())

    # Pooling Layer 1: halve the image size
    model.add(layers.MaxPooling2D((2, 2)))

    # Additional Convolutional and Pooling Layers
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and fully connected layer
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))

    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

# Load and preprocess the dataset
def load_data():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_data = datagen.flow_from_directory('dataset', target_size=(32, 32), batch_size=32, class_mode='binary', subset='training')
    val_data = datagen.flow_from_directory('dataset', target_size=(32, 32), batch_size=32, class_mode='binary', subset='validation')
    return train_data, val_data

# Compile and train the model
def train_model(model, train_data, val_data, epochs=5):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, validation_data=val_data, epochs=epochs)
    model.save('stop_sign_cnn.h5')

# Run the training process
if __name__ == "__main__":
    model = create_model()
    train_data, val_data = load_data()
    train_model(model, train_data, val_data)
