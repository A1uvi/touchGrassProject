import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
import glob 
import warnings
warnings.filterwarnings('ignore')

# Define constants
image_data = 'touchGrassProject/data/train'
batch_size = 8
target_size = (224, 224)
validation_split = 0.2

# Create Training Dataset
train = tf.keras.preprocessing.image_dataset_from_directory(
    image_data,
    validation_split=validation_split,
    subset="training",
    seed=100,
    image_size=target_size,
    batch_size=batch_size,
)

# Create Validation Dataset
validation = tf.keras.preprocessing.image_dataset_from_directory(
    image_data,
    validation_split=validation_split,
    subset="validation",
    seed=100,
    image_size=target_size,
    batch_size=batch_size,
)

# Store class names for later use
class_names = train.class_names
print(class_names)

# Define data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Apply data augmentation to the training dataset
train = train.map(lambda x, y: (data_augmentation(x, training=True), y))

# Create and compile the model
base_model = tf.keras.applications.MobileNetV3Large(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

keras_model1 = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(class_names), activation='softmax')  # Use class_names instead of train.class_names
])

checkpoint = ModelCheckpoint("my_keras_model.keras", save_best_only=True)
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
keras_model1.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
hist1 = keras_model1.fit(train, epochs=40, validation_data=validation, callbacks=[checkpoint, early_stopping])

# Evaluate the model on validation data
val_loss, val_accuracy = keras_model1.evaluate(validation)

# Print validation accuracy
print(f'Validation Accuracy: {val_accuracy:.4f}')

# Optional: Display classification report
y_true = []
y_pred = []

# Get true labels and predictions
for images, labels in validation:
    preds = keras_model1.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Generate classification report
print(classification_report(y_true, y_pred, target_names=class_names))

# Save the Keras model
keras_model1.save("touchGrassProject/models/flower_model.keras")  # Save in HDF5 format

# Evaluate the model on validation data
val_loss, val_accuracy = keras_model1.evaluate(validation)

# Print validation accuracy
print(f'Validation Accuracy: {val_accuracy:.4f}')

# Optional: Display classification report
y_true = []
y_pred = []

# Get true labels and predictions
for images, labels in validation:
    preds = keras_model1.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Generate classification report
print(classification_report(y_true, y_pred, target_names=class_names))