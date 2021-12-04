import streamlit as st
import json
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

n_classes = 120
model = models.inception_v3(pretrained=True)
model.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, n_classes))
model.load_state_dict(torch.load("inception_dog_breed.pt"))

class_to_idx = json.load(open("class_to_idx.json", "r"))
idx_to_class = {v: k for k, v in class_to_idx.items()}

def predict(path: str):
    """Predict from a given path"""
    img = Image.open(path)
    img = transform(img)
    model.eval()
    logits = model(img.unsqueeze(0))
    pred = logits.ravel().argmax().tolist()
    return pred


st.title("Upload + Classification Example")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    pred = predict(uploaded_file)
    st.write(f'Predicted label: {idx_to_class[pred]}')