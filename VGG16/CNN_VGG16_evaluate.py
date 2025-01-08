import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import random
import os

# Load the saved model
model = tf.keras.models.load_model("food_classifier_model.h5")

# Define paths
test_data_dir = '/Users/sylviamiller/Documents/MSML/FoodDetectionClassification/archive/food-101/food-101/test'

# Get the list of all class directories
class_dirs = [os.path.join(test_data_dir, cls) for cls in os.listdir(test_data_dir)]

# Get the list of all image paths
all_image_paths = []
all_labels = []
for class_dir in class_dirs:
    image_paths = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
    all_image_paths.extend(image_paths)
    all_labels.extend([os.path.basename(class_dir)] * len(image_paths))

# Combine image paths and labels
combined = list(zip(all_image_paths, all_labels))
random.shuffle(combined)
all_image_paths[:], all_labels[:] = zip(*combined)

# Create a reverse mapping
class_indices = {cls: idx for idx, cls in enumerate(sorted(set(all_labels)))}
reverse_mapping = {v: k for k, v in class_indices.items()}

# Randomly select 20 different classes
num_classes = len(class_indices)
random_classes = random.sample(range(num_classes), 20)

# Randomly select one image from each class
random_images = []
for class_idx in random_classes:
    class_images = [img_path for img_path, label in zip(all_image_paths, all_labels) if class_indices[label] == class_idx]
    random_image = random.choice(class_images)
    random_images.append(random_image)

# Display the random images with model estimated label and true label
for img_path in random_images:
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    predictions = model.predict(x)
    predicted_class = np.argmax(predictions[0])
    predicted_food_name = reverse_mapping[predicted_class]
    true_class = class_indices[os.path.basename(os.path.dirname(img_path))]
    true_food_name = reverse_mapping[true_class]
    
    plt.imshow(img)
    plt.title(f"Estimated Label: {predicted_food_name}\nTrue Label: {true_food_name}")
    plt.axis('off')
    plt.show()
