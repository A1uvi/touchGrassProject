import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Path to the new test image
img_path = 'touchGrassProject/data/train/california_poppy/40209652_6454993fe7_c.jpg'

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)  # Preprocess

# Display the image
plt.imshow(img)
plt.axis('off')
plt.show()

keras_model = tf.keras.models.load_model("touchGrassProject/models/flower_model_two.keras")
predictions = keras_model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# Map the predicted class index to the class name
cl_nm=['astilbe', 'bellflower', 'black_eyed_susan', 'calendula', 'california_poppy', 'carnation', 'common_daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip', 'water_lily']
class_name = cl_nm[predicted_class[0]]
print(f"The predicted class is: {class_name}")


