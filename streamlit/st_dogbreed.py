import io
from urllib.request import urlopen
import os.path as op
import streamlit as st
import json
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

class_to_idx = json.load(open("streamlit/class_to_idx.json", "r"))
idx_to_class = {v: k for k, v in class_to_idx.items()}
n_classes = len(idx_to_class.keys())  # number of breeds classes
model = models.inception_v3(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, n_classes)
)
# Note that I only save `fc` layer weights, and not the whole model.
# torch.save(model.fc.state_dict(), "fc.pt")
MODEL_PATH = "streamlit/fc.pt"
model.fc.load_state_dict(torch.load(MODEL_PATH))


def predict(path: str):
    """Predict from a given path"""
    img = Image.open(path)
    img = transform(img)
    model.eval()
    logits = model(img.unsqueeze(0))
    pred = logits.argmax().tolist()
    return pred


st.title("Dog Breed Classification")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    pred = predict(uploaded_file)
    st.write(f'Predicted label: {idx_to_class[pred].capitalize().replace("_", " ")}')
