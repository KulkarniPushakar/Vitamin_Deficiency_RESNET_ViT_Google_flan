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

