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

writer = SummaryWriter('runs/incremental_training_1')

# location from which the data will be downloaded:
data_path = 'training/home'
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
batch_size = 4
num_epochs = 1


class FilteredDataset(Dataset):
    def __init__(self, dataset, index, offset=0):
        # indices that we want in the Subset
        self.index = index
        # filtering of the dataset indices that we want in our subset
        self.original_indices = [i for i in range(len(dataset.targets)) if dataset.targets[i]==index]
        self.dataset = Subset(dataset, self.original_indices)
        # remapping degli indici con chiavi i valori degli indici del del task
        self.original2task = {index: offset + i for i in range(0, 1)}
        # remapping degli indici con chiavi i valori degli indici del del task
        #self.task2original = dict({(val, key) for (key, val) in self.original2task.items()})

    def __getitem__(self, idx):
        (x, y) = self.dataset[idx]
        return x, self.original2task[y]

    def __len__(self):
        return len(self.original_indices)


def train(net, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0.0
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
            if i % 500 == 499:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000), 'accuracy:', running_correct / 2000)
                # len(train_loader) = len(train_set)/batch_size = 50000/4
                running_loss = 0.0
                running_correct = 0.0


def class_accuracy(net, val_loader, classe):
    correct = 0
    total = 0
    with torch.no_grad():
        for pos, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                correct += c[i].item()
                total += labels.size(0)
    print('Accuracy of %5s : %2d %%' % (classes[classe], 100 * correct/ total))
    return correct/total



def main():

    # Use a standard transformation on all datasets.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Load the CIFAR10 training and test sets.
    train_set = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    # Create task datasets and data loaders for both training and testing.
    tasks = [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9)]

    # define the datasets and data loaders for training
    task_train_dss = [FilteredDataset(train_set, classes) for classes in tasks]
    task_train_dls = [DataLoader(ds, batch_size=batch_size) for ds in task_train_dss]

    # define the datasets and data loaders for validation
    task_test_dss = [FilteredDataset(test_set, classes) for classes in tasks]
    task_test_dls = [DataLoader(ds, batch_size=batch_size) for ds in task_test_dss]

    net = network.Net()
    criterion = nn.CrossEntropyLoss()
    # define a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    i = 0
    net.train()
    initial_accuracy = []
    for task in tasks:
        print(f'Network with class', i, classes[i])
        net.add_task(1)
        net.to(device)
        net.set_tasks([i])
        net.train()
        train(net, task_train_dls[i],criterion, optimizer, num_epochs)
        initial_accuracy.append(class_accuracy(net, task_test_dls[i], task))
        i= i+ 1


    #for i in range(len(initial_accuracy)):
        #writer.add_scalar("initial accuracy", initial_accuracy[i], i+1)

    final_accuracy = []
    i= 0
    for task in tasks:
        final_accuracy.append(class_accuracy(net, task_test_dls[i], task))
        i= i+1

    #for i in range(len(final_accuracy))_:
        #writer.add_scalar("final accuracy")
    x=[1,2,3,4,5,6,7,8,9,10]
    print('\n\nFINAL ACCURACY')
    plt.plot(x,initial_accuracy, color="red", marker="o")


    plt.plot(x, final_accuracy, color="blue", marker="o" )
    plt.show()



if __name__ == '__main__':
    main()