# Vitamin Deficiency Detection using Deep Learning
## Automated Dermatological & Oral Biomarker Analysis + Medical Chatbot

This project implements a non-invasive vitamin deficiency detection system using dermatological and oral images. It integrates:
1. A ResNet152V2 + Vision Transformer (ViT) hybrid model for image classification
2. A FLAN-T5–based medical chatbot for personalized nutritional guidance
3. A web-based frontend (HTML/CSS + Flask) for real-time interaction
4. The system predicts 11 vitamin deficiencies and provides severity levels, along with the ability to chat with an AI assistant for treatment, precautions, and nutrition advice.

# Model Architecture
##  ResNet152V2 + ViT Hybrid

1. ResNet extracts local & high-level features.
2. ViT extracts global relationships and attention patterns.
3. Combined to improve robustness on subtle skin/tongue changes.

# Key Features
## 1. Vitamin Deficiency Detection (Computer Vision Model)

Hybrid model: ResNet152V2 + ViT
Dataset: 6,600 dermatological and oral images
11 classification categories:
Vitamin A, B1, B2, B3, B9, B12, C, D, E, K
Zinc/Iron/Biotin/Protein deficiency
Achieved 87% overall accuracy
F1-score:

100% for Vitamin B2, B3, B9
>98% for Vitamin C, E, K
Lower performance for Vitamin A & D (due to subtle visual symptoms)

## 2. Medical Chatbot (NLP Model)

Powered by FLAN-T5 Base, fine-tuned on 1,500 nutrition & medical Q/A pairs

### Provides:
Causes of the deficiency
Symptoms
Precautions
Treatment & diet suggestions
Includes similarity-based retrieval using cosine similarity for accurate answers

## 3. Web Application Interface

Simple and responsive HTML/CSS UI
Upload images (skin, nails, tongue, lips.
View predictions + confidence score + severity levels
Chat with the medical assistant instantly
