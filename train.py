"""
Trains a PyTorch image classification model using device-agnostic code.
"""

from .maxvit import MaxViT, max_vit_tiny_224, max_vit_small_224, max_vit_base_224, max_vit_large_224
# import torch
# import cv2
# from helper_functions import set_seeds
# import numpy as np
# from torch import nn
# from  maxvit import maxvit 
# from torchvision import transforms
# import pandas as pd
# from transformers import AutoTokenizer 
# from transformers import AutoModelForSequenceClassification
# from scipy.special import softmax
# from nltk.sentiment import SentimentIntensityAnalyzer
# from tqdm.notebook import tqdm
# from torchinfo import summary
# import os
# import PIL
import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

import maxvit
model=maxvit.max_vit_tiny_224(num_classes=1000)

# Setup directories
train_dir = r'C:/Users/abhis/Music/dataset/train'
test_dir = r'C:/Users/abhis/Music/dataset/test'

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
# model = model_builder.TinyVGG(
#     input_shape=3,
#     hidden_units=HIDDEN_UNITS,
#     output_shape=len(class_names)
# ).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
