import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# ------------------------
# Flask app setup
# ------------------------
app = Flask(__name__)

# ------------------------
# Paths
# ------------------------
MODEL_PATH = r"E:\project_Vita\vitamin_resnet_vit_model.h5"
TRAIN_DIR = r"E:\project_Vita\output_dataset\train"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

T5_MODEL_PATH = r"E:\project_Vita\flan-t5-vitamin-model"

# ------------------------
# Custom layers for ResNet-ViT
# ------------------------
def reshape_patches(t): return tf.reshape(t, (-1, t.shape[1]*t.shape[2], t.shape[3]))
def global_avg_pool(t): return tf.reduce_mean(t, axis=1)

# ------------------------
# Load models
# ------------------------
print("📦 Loading ResNet-ViT model...")
model = load_model(MODEL_PATH, custom_objects={
    'reshape_patches': reshape_patches,
    'global_avg_pool': global_avg_pool
}, compile=False)

# ------------------------
# Load class labels from TRAIN_DIR
# ------------------------
CLASS_LABELS = sorted([
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
])
print(f"📂 Detected classes ({len(CLASS_LABELS)}): {CLASS_LABELS}")

# ------------------------
# Load T5 model
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_PATH)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_PATH).to(device)

# ------------------------
# Helper functions
# ------------------------
def predict_single_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    preds = model.predict(np.expand_dims(img_array, axis=0))
    return preds[0]

def generate_t5_response(deficiency, question):
    inputs = tokenizer(f"Deficiency: {deficiency}. Question: {question}", return_tensors="pt").to(device)
    outputs = t5_model.generate(**inputs, max_length=128, num_beams=4, temperature=0.8)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------------
# Routes
# ------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(img_path)

    probs = predict_single_image(img_path)
    idx = int(np.argmax(probs))

    # Safety check for index
    if idx >= len(CLASS_LABELS):
        return f"Error: predicted index {idx} out of range for class labels.", 500

    predicted_class = CLASS_LABELS[idx]
    confidence = probs[idx] * 100

    # Generate T5 outputs
    cause = generate_t5_response(predicted_class, f"What is the reason for {predicted_class}?")
    symptoms = generate_t5_response(predicted_class, f"What are the symptoms of {predicted_class}?")
    diet = generate_t5_response(predicted_class, f"What diet or remedies are recommended for {predicted_class}?")

    return render_template('result.html',
                           image_path=img_path,
                           deficiency=predicted_class,
                           confidence=f"{confidence:.2f}",
                           cause=cause,
                           symptoms=symptoms,
                           diet=diet)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']
    deficiency = request.form['deficiency']
    response = generate_t5_response(deficiency, user_input)
    return {'response': response}

# ------------------------
# Run app
# ------------------------
if __name__ == '__main__':
    app.run(debug=True)
