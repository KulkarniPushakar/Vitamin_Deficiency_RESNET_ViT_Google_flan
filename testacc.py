
#  checking result by taking rendom Images test dataset.

# import os
# import random
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import load_img, img_to_array
# import matplotlib.pyplot as plt
# from PIL import Image

# # ----------------------------
# # Custom Lambda functions used in model
# # ----------------------------
# def reshape_patches(t):
#     return tf.reshape(t, (-1, t.shape[1]*t.shape[2], t.shape[3]))

# def global_avg_pool(t):
#     return tf.reduce_mean(t, axis=1)

# # ----------------------------
# # Paths
# # ----------------------------
# MODEL_PATH = r"E:\project_Vita\vitamin_resnet_vit_model.h5"
# DATASET_DIR = r"E:\project_Vita\output_dataset\test"  # Folder that contains class folders

# # ----------------------------
# # Load model
# # ----------------------------
# model = load_model(
#     MODEL_PATH,
#     custom_objects={'reshape_patches': reshape_patches, 'global_avg_pool': global_avg_pool},
#     compile=False
# )
# print(f"✅ Model loaded successfully from: {MODEL_PATH}")

# # ----------------------------
# # Auto-detect class labels from folders
# # ----------------------------
# CLASS_LABELS = sorted([
#     d for d in os.listdir(DATASET_DIR)
#     if os.path.isdir(os.path.join(DATASET_DIR, d))
# ])
# print(f"📂 Detected {len(CLASS_LABELS)} classes: {CLASS_LABELS}")

# # ----------------------------
# # Pick random image from any class
# # ----------------------------
# def get_random_image(data_dir):
#     class_folder = random.choice(CLASS_LABELS)
#     class_path = os.path.join(data_dir, class_folder)
#     image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

#     if not image_files:
#         raise ValueError(f"No images found in {class_path}")

#     random_image = random.choice(image_files)
#     return os.path.join(class_path, random_image), class_folder

# # ----------------------------
# # Predict on a single image
# # ----------------------------
# def predict_single_image(model, img_path):
#     img = load_img(img_path, target_size=(224, 224))
#     img_array = img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     preds = model.predict(img_array)
#     return preds[0]

# # ----------------------------
# # Main
# # ----------------------------
# img_path, actual_class = get_random_image(DATASET_DIR)
# print(f"\n🖼️ Randomly selected image: {img_path}")

# probs = predict_single_image(model, img_path)
# predicted_index = np.argmax(probs)
# predicted_class = CLASS_LABELS[predicted_index]
# confidence = probs[predicted_index] * 100

# # ----------------------------
# # Results
# # ----------------------------
# print("\n" + "="*60)
# print("           🔍 RANDOM IMAGE PREDICTION RESULTS")
# print("="*60)
# print(f"Image Path:      {img_path}")
# print(f"Actual Class:    {actual_class}")
# print(f"Predicted Class: {predicted_class}")
# print(f"Confidence:      {confidence:.2f}%")
# print("="*60)

# # ----------------------------
# # Display the image
# # ----------------------------
# img = Image.open(img_path)
# plt.figure(figsize=(6,6))
# plt.imshow(img)
# plt.title(f"Actual: {actual_class}\nPredicted: {predicted_class} ({confidence:.2f}%)")
# plt.axis('off')
# plt.show()

#  code 2
#  Check resulte on the base of file path 
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------
# Custom Lambda functions used in model
# ----------------------------
def reshape_patches(t):
    return tf.reshape(t, (-1, t.shape[1]*t.shape[2], t.shape[3]))

def global_avg_pool(t):
    return tf.reduce_mean(t, axis=1)

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = r"E:\project_Vita\vitamin_resnet_vit_model.h5"
DATASET_DIR = r"E:\project_Vita\output_dataset\test"  # for class label reference

# ----------------------------
# Load model
# ----------------------------
model = load_model(
    MODEL_PATH,
    custom_objects={'reshape_patches': reshape_patches, 'global_avg_pool': global_avg_pool},
    compile=False
)
print(f"✅ Model loaded successfully from: {MODEL_PATH}")

# ----------------------------
# Auto-detect class labels from test dataset
# ----------------------------
CLASS_LABELS = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])
print(f"📂 Detected {len(CLASS_LABELS)} classes: {CLASS_LABELS}")

# ----------------------------
# Predict on a single external image
# ----------------------------
def predict_single_image(model, img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    return preds[0]

# ----------------------------
# Input image path manually
# ----------------------------
img_path = input("\nEnter the full path of the image to classify: ").strip()

if not os.path.exists(img_path):
    raise FileNotFoundError(f"❌ The file does not exist: {img_path}")

# ----------------------------
# Make prediction
# ----------------------------
print("\n🔍 Predicting...")
probs = predict_single_image(model, img_path)
predicted_index = np.argmax(probs)
predicted_class = CLASS_LABELS[predicted_index]
confidence = probs[predicted_index] * 100

# ----------------------------
# Results
# ----------------------------
print("\n" + "="*60)
print("            🧠 MODEL PREDICTION RESULTS")
print("="*60)
print(f"Image Path:      {img_path}")
print(f"Predicted Class: {predicted_class}")
print(f"Confidence:      {confidence:.2f}%")
print("="*60)

# ----------------------------
# Display the image
# ----------------------------
img = Image.open(img_path)
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%")
plt.axis('off')
plt.show()


