# ----------------------------
# ResNet152 + ViT Hybrid Model
# Compatible: TF 2.10.0, Python 3.10, NumPy 1.23.5
# ----------------------------

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Lambda
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# ----------------------------
# GPU configuration (optional)
# ----------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
print("Num GPUs Available: ", len(gpus))

# ----------------------------
# Dataset paths
# ----------------------------
base_dir = input("Enter the full path to your dataset folder (train/test inside): ").strip()
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Image & batch size
img_size = (224, 224)
batch_size = 8

# ----------------------------
# Data Generators
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

num_classes = len(train_gen.class_indices)
print(f"Number of classes detected: {num_classes}")

# ----------------------------
# Load ResNet152V2 Base Model
# ----------------------------
base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

resnet_output = base_model.output
print("ResNet output shape:", resnet_output.shape)

# ----------------------------
# Convert to patches (ViT tokens)
# ----------------------------
x = Lambda(lambda t: tf.reshape(t, (-1, t.shape[1]*t.shape[2], t.shape[3])),
           output_shape=(49, 2048))(resnet_output)

# ----------------------------
# Transformer Encoder
# ----------------------------
dim = 2048
num_heads = 4
ff_dim = 1024

# Layer Norm
x_norm = LayerNormalization(epsilon=1e-6)(x)

# Multi-Head Self Attention
attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=dim//num_heads)(x_norm, x_norm)
x = attn_output + x  # Residual

# Feed Forward
x_norm2 = LayerNormalization(epsilon=1e-6)(x)
ff = Dense(ff_dim, activation='relu')(x_norm2)
ff = Dense(dim)(ff)
x = ff + x  # Residual

# Global Average Pooling
x = Lambda(lambda t: tf.reduce_mean(t, axis=1),
           output_shape=(2048,))(x)

# Dropout & Classifier
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

# ----------------------------
# Create Model
# ----------------------------
model = Model(inputs=base_model.input, outputs=output)
# model.summary()

# ----------------------------
# Compile Model
# ----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------
# Train Model
# ----------------------------
epochs = 10
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=epochs
)

# ----------------------------
# Evaluate Model
# ----------------------------
loss, acc = model.evaluate(test_gen)
print(f"\nTest Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# ----------------------------
# Save Model
# ----------------------------
model.save("vitamin_resnet_vit_model_1.h5")
print("Model saved as 'vitamin_resnet_vit_model_1.h5'")

# ----------------------------
# Plot Accuracy & Loss
# ----------------------------
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
