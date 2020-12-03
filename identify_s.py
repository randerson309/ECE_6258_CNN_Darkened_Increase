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

    model = torch.load('/Users/cyb/model_sign.pt')

    transformations = transforms.Compose([transforms.Resize([45, 45]), transforms.ToTensor()])

    iden = datasets.ImageFolder('/Users/cyb/isign', transform = transformations)

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


    s1_des = "/Users/cyb/SIGN/1"
    s2_des = "/Users/cyb/SIGN/2"
    s3_des = "/Users/cyb/SIGN/3"
    s4_des = "/Users/cyb/SIGN/4"
    s5_des = "/Users/cyb/SIGN/5"
    s6_des = "/Users/cyb/SIGN/6"
    s7_des = "/Users/cyb/SIGN/7"
    s8_des = "/Users/cyb/SIGN/8"


    # Use it only when the directory doesn't exist
    """
    os.makedirs(dark_des)
    os.makedirs(norm_des)
    """

    # Change loop iterations based on how many to sort
    for x in range(2447):
        
        
        if classes[x].item() == 1:
            shutil.move(i_loader.dataset.samples[x][0], s1_des)
        elif classes[x].item() == 2:
            shutil.move(i_loader.dataset.samples[x][0], s2_des)
        elif classes[x].item() == 3:
            shutil.move(i_loader.dataset.samples[x][0], s3_des)
        elif classes[x].item() == 4:
            shutil.move(i_loader.dataset.samples[x][0], s4_des)
        elif classes[x].item() == 5:
            shutil.move(i_loader.dataset.samples[x][0], s5_des)
        elif classes[x].item() == 6:
            shutil.move(i_loader.dataset.samples[x][0], s6_des)
        elif classes[x].item() == 7:
            shutil.move(i_loader.dataset.samples[x][0], s7_des)
        else:
            shutil.move(i_loader.dataset.samples[x][0], s8_des)
        




if __name__ == '__main__':
    main()
