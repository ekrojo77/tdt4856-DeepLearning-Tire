from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.autograd import Variable
import skimage
from skimage import io
from PIL import Image
import requests
import io
import torchvision.transforms.functional as TF
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 1

data_dir = "validation-images"

local_model = "1584524997\model-17.pt"

model_ft = models.resnet18()
#set_parameter_requires_grad(model_ft, feature_extract)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)
input_size = 224

def load_model(model, PATH):
    model.load_state_dict(torch.load(os.path.abspath('')+'\Models\\'+PATH))
    model.eval()
    return model


# HERE YOU SET YOUR LOCAL MODEL
model_loaded = load_model(model_ft, local_model )

def load_image(PATH):
    #InputImg = skimage.img_as_float(skimage.io.imread(PATH))
    image_datasets = datasets.ImageFolder(root = data_dir, transform=data_transforms['train'])
    return image_datasets
'''
def transform( aImage ):
    ptLoader = transforms.Compose([transforms.ToTensor()])
    aImage = ptLoader( aImage ).float()
    aImage = Variable( aImage, volatile=True  )
    return aImage.cuda()
'''
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train']}

class_names = image_datasets['train'].classes



def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

totalAttempts = 0
success = 0
for inputs, labels in dataloaders_dict['train']:
    
    print("_____________________________________________")
    print("\n")
    prediction = model_loaded(inputs)
    resultArray = prediction.detach().numpy()[0]
    if(resultArray[0]>resultArray[1]):
        predLabel = 0
    if(resultArray[1]>resultArray[0]):
        predLabel = 1
    print("Predicting the label: ",class_names[predLabel])
    if(labels.numpy()[0] == predLabel):
        print("Correct prediction in regards to label set at data creation")
        success += 1
    for i in range(0,batch_size):
        imshow(inputs[i])
        
    totalAttempts += 1
    
    
print("_____________________________________________")
print("\n")
print("Number of images: ", totalAttempts)
print("Number of correct predictions: ", success)
print("success: ", success/totalShit*100,"%")
print("_____________________________________________")