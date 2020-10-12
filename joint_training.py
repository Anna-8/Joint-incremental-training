import network
import torch
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, Subset
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

writer1 = SummaryWriter('runs/joint_training1')
writer2 = SummaryWriter('runs/joint_training2')

writer3 = SummaryWriter('runs/joint_training3')

writer4 = SummaryWriter('runs/joint_training4')
writer5 = SummaryWriter('runs/joint_training5')
writers= [writer1, writer2, writer3, writer4, writer5]
# location from which the data will be downloaded:
data_path = 'training/home'
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
batch_size = 4
num_epochs = 5


def train(net, train_loader, criterion, optimizer,  epochs):
    x=[2000,4000,6000,8000,10000,12000]


    for epoch in range(epochs):
        writer= writers[epoch]
        running_loss = 0.0
        running_correct = 0.0
        total = 0.0
        training_loss = []
        accuracy = []
        for i, (inputs, labels) in enumerate(train_loader):
            # i is the data position counting in this case from 0
            # move inputs and labels to the device we are training on
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            # computes a value that estimates how far away the output is from the target.
            loss = criterion(outputs, labels)
            # compute the gradients of the loss with respect to all parameters
            loss.backward()
            # updates the model
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # returns the max value reduce to dimension 1
            running_correct += (predicted == labels).sum().item()
            total += labels.size(0)
            if i % 2000 == 1999 :  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % ( epoch + 1, i+1 , running_loss / 2000), 'accuracy:', running_correct / 8000)
                training_loss.append(running_loss/2000)
                accuracy.append(running_correct/total)
                writer.add_scalar('training loss', running_loss / 2000, global_step=len(train_loader)+i)
                # writer.add_scalar('accuracy', running_correct / 2000, global_step=epoch*len(train_loader)+i)
                running_loss = 0.0
                running_correct = 0.0
                total=0
                writer.close()




def accuracy(net, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def class_accuracy(net, test_loader):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


def main():
    # Use a standard transformation on all datasets.
    transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    tasks = [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)]
    # Create task datasets and data loaders for both training and testing
    # CIFAR10 provides 50000 samples for training and 10000 for testing
    train_set = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    net = network.Net()
    net.add_task(10)
    # move the net to the device we are training on
    net.to(device)
    net.set_tasks([0])

    # sets the module in training mode
    # has any effect only on certain modules like dropout, which behave different
    net.train()
    # define the loss function, that compare the outputs of the net to the desired output
    criterion = nn.CrossEntropyLoss()
    # define a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train(net, train_loader, criterion, optimizer, num_epochs)
    net.eval()
    accuracy(net,test_loader)
    class_accuracy(net,test_loader)


if __name__ == '__main__':
    main()