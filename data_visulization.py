# ============================================================
# 📊 Vitamin Deficiency Classification — Correct Evaluation & Visualization
# ============================================================

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # ----------------------------
# # 🧩 Custom Lambda functions (used in your ResNet + ViT model)
# # ----------------------------
# def reshape_patches(t):
#     return tf.reshape(t, (-1, t.shape[1] * t.shape[2], t.shape[3]))

# def global_avg_pool(t):
#     return tf.reduce_mean(t, axis=1)

# # ----------------------------
# # 📁 Paths
# # ----------------------------
# MODEL_PATH = r"E:\project_Vita\vitamin_resnet_vit_model_1.h5"
# TRAIN_DIR = r"E:\project_Vita\output_dataset\train"
# TEST_DIR = r"E:\project_Vita\output_dataset\test"

# # ----------------------------
# # 🚀 Load Model
# # ----------------------------
# model = load_model(
#     MODEL_PATH,
#     custom_objects={'reshape_patches': reshape_patches, 'global_avg_pool': global_avg_pool},
#     compile=False
# )
# print(f"✅ Model loaded successfully from: {MODEL_PATH}")

# # ----------------------------
# # 📋 Load Class Labels
# # ----------------------------
# CLASS_LABELS = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
# print(f"📂 Detected Classes: {CLASS_LABELS}")

# # ----------------------------
# # 🧪 Load Test Dataset
# # ----------------------------
# img_size = (224, 224)
# batch_size = 16

# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     TEST_DIR,
#     image_size=img_size,
#     batch_size=batch_size,
#     shuffle=False
# )

# # ----------------------------
# # ✅ Extract True Labels (Entire Test Set)
# # ----------------------------
# y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)

# # ----------------------------
# # 🔮 Predict on Test Set (Entire Dataset)
# # ----------------------------
# y_pred_probs = model.predict(test_ds)
# y_pred = np.argmax(y_pred_probs, axis=-1)

# # ----------------------------
# # 📈 Metrics — Entire Test Set
# # ----------------------------
# print("\n📊 Classification Report (Entire Test Set):\n")
# print(classification_report(y_true, y_pred, target_names=CLASS_LABELS, digits=4))

# acc = accuracy_score(y_true, y_pred)
# prec = precision_score(y_true, y_pred, average='weighted')
# rec = recall_score(y_true, y_pred, average='weighted')
# f1 = f1_score(y_true, y_pred, average='weighted')

# print(f"\n✅ Accuracy:  {acc:.4f}")
# print(f"✅ Precision: {prec:.4f}")
# print(f"✅ Recall:    {rec:.4f}")
# print(f"✅ F1-Score:  {f1:.4f}")

# # ============================================================
# # 🖼️ Visualization Section — Separate for Single Images
# # ============================================================

# # 1️⃣ Sample Images from Each Class
# plt.figure(figsize=(15, 10))
# for i, cls in enumerate(CLASS_LABELS[:9]):
#     cls_path = os.path.join(TRAIN_DIR, cls)
#     img_file = random.choice(os.listdir(cls_path))
#     img_path = os.path.join(cls_path, img_file)
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(load_img(img_path))
#     plt.title(cls)
#     plt.axis('off')
# plt.suptitle("🖼️ Sample Images from Vitamin Deficiency Classes", fontsize=16)
# plt.show()

# # 2️⃣ Class Distribution
# class_counts = {cls: len(os.listdir(os.path.join(TRAIN_DIR, cls))) for cls in CLASS_LABELS}
# sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
# plt.xticks(rotation=45)
# plt.title("📦 Number of Images per Class (Training Data)")
# plt.ylabel("Image Count")
# plt.xlabel("Vitamin Deficiency Class")
# plt.show()

# # 3️⃣ Confusion Matrix — Entire Test Set
# cm = confusion_matrix(y_true, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
# disp.plot(cmap='Purples', xticks_rotation=45)
# plt.title("🧠 Confusion Matrix for Vitamin Deficiency Classification")
# plt.show()

# # 4️⃣ Grad-CAM Visualization — Single Random Test Image
# def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
#     grad_model = tf.keras.models.Model(
#         [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
#     )
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         if pred_index is None:
#             pred_index = tf.argmax(predictions[0])
#         class_channel = predictions[:, pred_index]
#     grads = tape.gradient(class_channel, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]
#     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
#     heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
#     return heatmap.numpy()

# # Choose a single random test image
# sample_img_path = random.choice([
#     os.path.join(TEST_DIR, cls, random.choice(os.listdir(os.path.join(TEST_DIR, cls))))
#     for cls in CLASS_LABELS
# ])
# img = load_img(sample_img_path, target_size=img_size)
# img_array = img_to_array(img) / 255.0
# img_array = np.expand_dims(img_array, axis=0)

# # Grad-CAM heatmap
# heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv5_block3_out')
# plt.matshow(heatmap)
# plt.title(f"🔥 Grad-CAM Heatmap for {os.path.basename(sample_img_path)}")
# plt.show()

# # 5️⃣ Confidence Distribution for the Same Single Image
# pred_probs = model.predict(img_array)[0]
# plt.figure(figsize=(10, 5))
# plt.bar(CLASS_LABELS, pred_probs * 100)
# plt.xticks(rotation=45)
# plt.title(f"📊 Confidence Distribution for {os.path.basename(sample_img_path)}")
# plt.ylabel("Confidence (%)")
# plt.show()

# print("\n✅ Evaluation & Visualization Completed Successfully!")
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()
