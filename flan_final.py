import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def reshape_patches(t):
    return tf.reshape(t, (-1, t.shape[1]*t.shape[2], t.shape[3]))

def global_avg_pool(t):
    return tf.reduce_mean(t, axis=1)

MODEL_PATH = r"E:\project_Vita\vitamin_resnet_vit_model.h5"
DATASET_DIR = r"E:\project_Vita\output_dataset\test"
T5_MODEL_PATH = r"E:\project_Vita\flan-t5-vitamin-model"

print("📦 Loading ResNet-ViT model...")
model = load_model(
    MODEL_PATH,
    custom_objects={'reshape_patches': reshape_patches, 'global_avg_pool': global_avg_pool},
    compile=False
)
print(f"✅ Model loaded successfully from: {MODEL_PATH}")

CLASS_LABELS = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])
print(f"📂 Detected {len(CLASS_LABELS)} classes: {CLASS_LABELS}")

print("\n💬 Loading fine-tuned FLAN-T5 model...")
tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_PATH)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_PATH)
print("✅ FLAN-T5 Vitamin Chatbot model loaded successfully!")

def predict_single_image(model, img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    return preds[0]

def generate_t5_response(deficiency, question):
    prompt = f"Deficiency: {deficiency}. Question: {question}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    t5_model.to(device)
    outputs = t5_model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        temperature=0.8
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n💬 Welcome to the Vitamin Deficiency Assistant!")
print("📸 Please provide an image path to predict the vitamin deficiency.\n")

img_path = input("🖼️ Enter the full path of the image: ").strip()
if not os.path.exists(img_path):
    raise FileNotFoundError(f"❌ The file does not exist: {img_path}")

print("\n🔍 Predicting deficiency...")
probs = predict_single_image(model, img_path)
predicted_index = np.argmax(probs)
predicted_class = CLASS_LABELS[predicted_index]
confidence = probs[predicted_index] * 100

print("\n" + "="*60)
print("            🧠 MODEL PREDICTION RESULTS")
print("="*60)
print(f"Image Path:      {img_path}")
print(f"Predicted Class: {predicted_class}")
print(f"Confidence:      {confidence:.2f}%")
print("="*60)

print(f"\n💡 Details for {predicted_class}:")

reason = generate_t5_response(predicted_class, f"What is the reason for {predicted_class}?")
symptoms = generate_t5_response(predicted_class, f"What are the symptoms of {predicted_class}?")
diet = generate_t5_response(predicted_class, f"What diet or remedies are recommended for {predicted_class}?")

print("\n🧠 Comprehensive Report:\n")
print(
    f"For {predicted_class}, the primary cause is {reason}. "
    f"Common symptoms include {symptoms}. "
    f"To manage or recover from {predicted_class}, a recommended diet or remedy is {diet}."
)
print("\n💬 You can now chat about this deficiency. Type 'exit' to quit.")
while True:
    user_input = input("🧠 You: ").strip()
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("👋 Exiting chat. Stay healthy!")
        break
    response = generate_t5_response(predicted_class, user_input)
    print(f"🤖 Bot: {response}")

img = Image.open(img_path)
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%")
plt.axis('off')
plt.show()
