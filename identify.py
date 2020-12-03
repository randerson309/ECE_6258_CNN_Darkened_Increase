import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

def main():

    model = torch.load('/Users/cyb/model2.pt')

    transformations = transforms.Compose([transforms.Resize([45, 45]), transforms.ToTensor()])

    iden = datasets.ImageFolder('/Users/cyb/TO_IDENTIFY', transform = transformations)

    i_loader = torch.utils.data.DataLoader(iden, batch_size=1000, shuffle=False)

    device = torch.device("cpu")

    model.eval()

    

    with torch.no_grad():
        for inputs, labels in i_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            output = model.forward(inputs)

            output = torch.exp(output)

            probs, classes = output.topk(1, dim=1)


    dark_des = "/Users/cyb/IDENTIFIED/dark"
    norm_des = "/Users/cyb/IDENTIFIED/norm"


    # Use it only when the directory doesn't exist
    """
    os.makedirs(dark_des)
    os.makedirs(norm_des)
    """

    # Change loop iterations based on how many to sort
    for x in range(2447):
        
        
        if classes[x].item() == 0:
            shutil.move(i_loader.dataset.samples[x][0], dark_des)
        else:
            shutil.move(i_loader.dataset.samples[x][0], norm_des)
        




if __name__ == '__main__':
    main()
