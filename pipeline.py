import torch
import maxvit
import cv2
from helper_functions import set_seeds
import numpy as np
from torch import nn
from torchvision import transforms
import pandas as pd
from transformers import AutoTokenizer 
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from torchinfo import summary
import os
import PIL
from PIL import Image

# Load MaxViT model
network = maxvit.max_vit_tiny_224(num_classes=1000)
# csv_file_path = 'IMAGENET.csv'
# df = pd.read_csv(csv_file_path)
# label_arr=[]


# Define preprocessing transformations
preprocess = transforms.Compose([
        transforms.Resize((224, 224)),                            
         transforms.ToPILImage(),
       transforms.ToTensor(),                                     
      transforms.Normalize(mean=[0.485, 0.456, 0.406],            
                         std=[0.229, 0.224, 0.225])
       
        # transforms.Normalize(img_mean, img_std),
        # lambda x: np.rollaxis(x.numpy(), 0, 3)
    ])
# self.transform = transforms.Compose([])

train_dir = r'C:/Users/abhis/Music/dataset/train'
test_dir = r'C:/Users/abhis/Music/dataset/test'

for parameter in network.parameters():
    parameter.requires_grad = True

class_names = ['safe','unsafe']

set_seeds()
network.heads = nn.Linear(in_features=48, out_features=len(class_names)).to(device="cpu")    


# Print a summary using torchinfo (uncomment for actual output)
summary(model=network, 
        input_size=(16, 3, 224, 224), # (batch_size, color_channels, height, width)
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)


pretrained_transforms = preprocess
# print(pretrained_transforms)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    # num_workers: int=NUM_WORKERS
    
):


  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
    #   num_workers=num_workers,
      pin_memory=True,
    #   persistent_workers=True
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
    #   num_workers=num_workers,
    #   persistent_workers=True,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names

train_dataloader_pretrained, test_dataloader_pretrained, class_names =create_dataloaders(train_dir=train_dir,test_dir=test_dir,transform=pretrained_transforms,batch_size=16) 

from going_modular.going_modular import engine
from predictions import pred_and_plot_image  

# Create optimizer and loss function
optimizer = torch.optim.Adam(params=network.parameters(), 
                             lr=1e-3)
# loss_fn = torch.nn.CrossEntropyLoss()
# torch.set_grad_enabled(True) 
# loss = loss_fn(output, torch.tensor([[10.0,31.0]]).double()).float()
 # Context-manager 
# Train the classifier head of the pretrained ViT feature extractor model
# set_seeds()
# main_model= engine.train(model=network,
#                                       train_dataloader=train_dataloader_pretrained,
#                                       test_dataloader=test_dataloader_pretrained,
#                                       optimizer=optimizer,
#                                       loss_fn=torch.nn.CrossEntropyLoss(),
#                                       epochs=10,
#                                       device="cpu")
# pretrained_vit_results.save_pretrained(r"C:\Users\abhis\Music\video understanding\MaxViT-master\maxvit\model.py")
# torch.save(main_model,r'C:\Users\abhis\Music\video understanding\MaxViT-master\maxvit\model.pt')
#  torch.save(optimizer.state_dict(),r'C:\Users\abhis\Music\video understanding\MaxViT-master\maxvit\optimizer.pt')


# Move the model to the desired device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # from helper_functions import plot_loss_curves

# import ast

# # Load the encoded text from file
# with open('maxvit\model_Saved', 'r') as f:
#     encoded_text = f.read()

# # Decode the encoded text into a dictionary object
# state_dict = ast.literal_eval(encoded_text)

my_model=(torch.load(r'C:\Users\abhis\Music\video understanding\MaxViT-master\maxvit\model.pt'))
#  main_model.load_state_dict(torch.load(r'C:\Users\abhis\Music\video understanding\MaxViT-master\maxvit\optimizer.pt'))
# main_model = main_model.to(device)

# print(type(main_model))

# # Plot our ViT model's loss curves
# plot_loss_curves(pretrained_vit_results)

# import requests


# # Import function to make predictions on images and plot them 
# from going_modular.going_modular.predictions import pred_and_plot_image

# # # Setup custom image path
from PIL import Image
custom_image_path =r"C:\Users\abhis\Music\video understanding\MaxViT-master\maxvit\1.jpg"
image=Image.open(custom_image_path)

import torch
import torchvision.transforms as transforms

transform=transforms.ToTensor()
img=transform(image)
input=preprocess(img).unsqueeze(0)
# print(type(input))
# output2=network(custom_image_path)
# print(output2)

import torchvision.models as models
my_model = models.resnet18(pretrained=True)
my_model.eval()

with torch.no_grad():
    output = my_model(input)
    output2=network(input)

pred_probs = torch.softmax(output2, dim=1)
pred_label = torch.argmax(pred_probs, dim=1)

# print(type(pred_label))
# print(type(pred_probs))
print(class_names[pred_label]) 
# print(pred_probs)
# print(class_names[pred_label])
# print(pred_probs.max())
# print(output)
# Predict on custom image
# pred_and_plot_image(model=my_model,
#                     image_path=custom_image_path,
#                     class_names=class_names)

# self.transform = transforms.Compose([transforms.ToTensor()])

# Open video capture

# while not cap.isOpened():
#     cap = cv2.VideoCapture("deer.avi")
#     cv2.waitKey(1000)
#     print ("Wait for the header")


# pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
# while True:
#     flag, frame = cap.read()
#     if flag:
#         # The frame is ready and already captured
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Preprocess frame
#         input_image = preprocess(frame_rgb).unsqueeze(0)  # Add batch dimension

#     # Generate predictions
#         with torch.no_grad():
#             output = network(input_image)

#     # Post-process the output
#         probabilities = torch.nn.functional.softmax(output, dim=1)
#         predicted_class_index = torch.argmax(probabilities, dim=1)

#     # Print prediction for demonstration
#         print("Predicted class index:", predicted_class_index.item())
#         # cv2.imshow('video', frame)
#         pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
#         print (str(pos_frame)+" frames")
#     else:
#         # The next frame is not ready, so we try to read it again
#         cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
#         print ("frame is not ready")
#         # It is better to wait for a while for the next frame to be ready
#         cv2.waitKey(1000)

#     if cv2.waitKey(10) == 27:
#         break
#     if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
#         # If the number of captured frames is equal to the total number of frames,
#         # we stop
#         break


# cap = cv2.VideoCapture('summary_video.avi')

# # Preprocess frame
# while cap.isOpened():
#     # Read frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to RGB (MaxViT expects RGB images)
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
#     # Preprocess frame
#     input_image = preprocess(frame_rgb).unsqueeze(0)  # Add batch dimension
      
#     # Generate predictions
#     with torch.no_grad():
#         output = network(input_image) 
   
#     # Post-process the output
#     probabilities = torch.nn.functional.softmax(output, dim=1)
#     predicted_class_index = torch.argmax(probabilities, dim=1)

    # Print prediction for demonstration
    
    # print("Predicted class index:", predicted_class_index.item())
    

    # Assuming your CSV file has columns 'ClassIndex' and 'ClassLabel'
    # Replace 'path_to_your_csv_file.csv' with the actual path to your CSV file
    

    # Load the CSV file into a pandas DataFrame
    

    # Assuming class_index is the variable containing the class index you want to look up
    # class_id = 4  # Replace 1 with your actual class index

    # Filter the DataFrame to get the row corresponding to the given class index
    # class_row = df[df['Class ID'] == predicted_class_index.item()]
    # # print(class_row)
    # # Get the class label from the filtered row
    # class_label = class_row['Class Name'].values[0] if not class_row.empty else None

    # Print the corresponding class label
    # print("Class Label:", class_label) 
    # label_arr.append(class_label)
    # Display prediction on frame (for visualization)
    # cv2.putText(frame, f"Prediction: {predicted_class_index.item()}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break

# Release video capture and close windows
# cap.release()
# cv2.destroyAllWindows()
# print(label_arr)

# example = ' '.join(label_arr)

# MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# # example="this is love"
# print(example)
# sia = SentimentIntensityAnalyzer()
# sia.polarity_scores(example)

# def polarity_scores_roberta(example):
#     encoded_text = tokenizer(example, return_tensors='pt')
#     output = model(**encoded_text)
#     scores = output[0][0].detach().numpy()
#     scores = softmax(scores)
#     scores_dict = {
#         'roberta_neg' : scores[0],
#         'roberta_neu' : scores[1],
#         'roberta_pos' : scores[2]
#     }
#     return scores_dict

# roberta_result = polarity_scores_roberta(example)
# print(roberta_result)
