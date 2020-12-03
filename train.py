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



def main():

    transformations = transforms.Compose([transforms.Resize([45, 45]), transforms.ToTensor()])

    train_set = datasets.ImageFolder('/Users/cyb/TSR_DATA/train', transform = transformations)
    test_set = datasets.ImageFolder('/Users/cyb/TSR_DATA/test', transform = transformations)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1000, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=True)

    model = models.densenet161(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier_input = model.classifier.in_features

    num_labels = 2

    classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))

    model.classifier = classifier

    device = torch.device("cpu")

    model.to(device)

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters())

    epochs = 10
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        accuracy = 0
        
        # Training the model
        model.train()
        counter = 0
        for inputs, labels in train_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Clear optimizers
            optimizer.zero_grad()
            # Forward pass
            output = model.forward(inputs)
            # Loss
            loss = criterion(output, labels)
            # Calculate gradients (backpropogation)
            loss.backward()
            # Adjust parameters based on gradients
            optimizer.step()
            # Add the loss to the training set's rnning loss
            train_loss += loss.item()*inputs.size(0)
            
            # Print the progress of our training
            counter += 1
            print(counter, "/", len(train_loader))
            
        # Evaluating the model
        model.eval()
        counter = 0
        # Tell torch not to calculate gradients
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Move to device
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass
                output = model.forward(inputs)
                # Calculate Loss
                valloss = criterion(output, labels)
                # Add loss to the validation set's running loss
                val_loss += valloss.item()*inputs.size(0)
                
                # Since our model outputs a LogSoftmax, find the real 
                # percentages by reversing the log function
                output = torch.exp(output)
                # Get the top class of the output
                top_p, top_class = output.topk(1, dim=1)
                # See how many of the classes were correct?
                equals = top_class == labels.view(*top_class.shape)
                # Calculate the mean (get the accuracy for this batch)
                # and add it to the running accuracy for this epoch
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                # Print the progress of our evaluation
                counter += 1
                print(counter, "/", len(test_loader))
        
        # Get the average loss for the entire epoch
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = val_loss/len(test_loader.dataset)
        # Print out the information
        print('Accuracy: ', accuracy/len(test_loader))
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))


    torch.save(model, '/Users/cyb/model2_new.pt')


if __name__ == '__main__':
    main()
