import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import glob 
import warnings

warnings.filterwarnings('ignore')

# Load dataset
df_train = 'touchGrassProject/data/train'
df_test = 'touchGrassProject/data/val'

# Check file counts in training and test directories
print(f"Training files: {len(os.listdir(df_train))}")
print(f"Validation files: {len(os.listdir(df_test))}")

# Prepare image files and labels
files = [i for i in glob.glob(df_train + "//*//*")]
np.random.shuffle(files)
labels = [os.path.dirname(i).split("/")[-1] for i in files]
data = zip(files, labels)
dataframe = pd.DataFrame(data, columns=["Image", "Label"])

# Visualize class distribution
plt.figure(figsize=(23,5))
sns.countplot(x=dataframe["Label"])
plt.title('Class Distribution')
plt.xticks(rotation=45)
plt.show()

# Parameters
batch_size = 32
target_size = (224, 224)
validation_split = 0.2

# Data augmentation with additional techniques
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),  # Added contrast adjustment
    tf.keras.layers.RandomBrightness(0.2),  # Added brightness adjustment
])

# Prepare training and validation datasets
train = tf.keras.preprocessing.image_dataset_from_directory(
    df_train,
    validation_split=validation_split,
    subset="training",
    seed=100,
    image_size=target_size,
    batch_size=batch_size,
)

validation = tf.keras.preprocessing.image_dataset_from_directory(
    df_train,
    validation_split=validation_split,
    subset="validation",
    seed=200,
    image_size=target_size,
    batch_size=batch_size,
)

# Check class names
cl_nm = train.class_names
print(cl_nm)
'''
# Build model
base_model = tf.keras.applications.MobileNetV3Large(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-20:]:  # Unfreeze the last 20 layers
    layer.trainable = True

# Create the model with L2 regularization
keras_model = tf.keras.models.Sequential([
    data_augmentation,  # Add data augmentation
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),  # Existing dropout
    tf.keras.layers.Dense(len(cl_nm), activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001))  # Added L2 regularization
])

# Compile the model with a smaller learning rate
keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),  # Reduced learning rate
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# Model summary
keras_model.summary()

# Callbacks
checkpoint = ModelCheckpoint("touchGrassProject/models/flower_model_two.keras", save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy', mode='max')

# Train the model
history = keras_model.fit(train, validation_data=validation, epochs=30, callbacks=[checkpoint, early_stopping])

# Load test dataset
test = tf.keras.preprocessing.image_dataset_from_directory(
    df_test,
    image_size=(224, 224),
    batch_size=batch_size,
    shuffle=False
)

checkpoint = ModelCheckpoint("touchGrassProject/models/flower_model_two.keras", save_best_only=True, monitor='val_loss', mode='min')
print('here')

# Predict on the test dataset
predictions = keras_model.predict(test)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels from the test dataset
true_labels = np.concatenate([y.numpy() for _, y in test], axis=0)

# Calculate accuracy
accuracy = np.mean(predicted_classes == true_labels)
print(f"Accuracy: {accuracy:.2f}")

# Create confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=cl_nm, yticklabels=cl_nm)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Classification report
class_report = classification_report(true_labels, predicted_classes, target_names=cl_nm)
print(class_report)
'''

# accuracy: 0.96
'''

                  precision    recall  f1-score   support

         astilbe       1.00      1.00      1.00         7
      bellflower       1.00      1.00      1.00         7
black_eyed_susan       1.00      1.00      1.00         7
       calendula       1.00      1.00      1.00         7
california_poppy       1.00      0.86      0.92         7
       carnation       1.00      0.57      0.73         7
    common_daisy       0.88      1.00      0.93         7
       coreopsis       0.88      1.00      0.93         7
       dandelion       1.00      1.00      1.00         7
            iris       1.00      1.00      1.00         7
            rose       1.00      1.00      1.00         7
       sunflower       1.00      1.00      1.00         7
           tulip       1.00      1.00      1.00         7
      water_lily       0.78      1.00      0.88         7

        accuracy                           0.96        98
       macro avg       0.97      0.96      0.96        98
    weighted avg       0.97      0.96      0.96        98
    '''