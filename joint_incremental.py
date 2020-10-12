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

writer = SummaryWriter('runs/incremental_training')
w = SummaryWriter('runs/incremental_training2')

# location from which the data will be downloaded:
data_path = 'training/home'
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
batch_size = 4
num_epochs = 1


class FilteredDataset(Dataset):
    def __init__(self, dataset, indices, offset=0):
        # indices that we want in the Subset
        self.indices = indices
        # filtering of the dataset indices that we want in our subset
        self.original_indices = [i for i in range(len(dataset.targets)) if dataset.targets[i] in indices]
        self.dataset = Subset(dataset, self.original_indices)
        # remapping degli indici con chiavi i valori degli indici del del task
        self.original2task = {indices[i]: offset + i for i in range(0, len(indices))}
        # remapping degli indici con chiavi i valori degli indici del del task
        self.task2original = dict({(val, key) for (key, val) in self.original2task.items()})

    def __getitem__(self, idx):
        (x, y) = self.dataset[idx]
        return x, self.original2task[y]

    def __len__(self):
        return len(self.original_indices)


def train(net, train_loader, task, criterion, optimizer, epochs):
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
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500), 'accuracy:',
                      running_correct / 500)
                # len(train_loader) = len(train_set)/batch_size = 50000/4
                running_loss = 0.0
                running_correct = 0.0


def task_accuracy(net, val_loader, task):
    class_correct = list(0. for i in range(len(task)))
    class_total = list(0. for i in range(len(task)))
    with torch.no_grad():
        for pos, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    avarage_accuracy = ((class_correct[0] / class_total[0]) + (class_correct[1] / class_total[1]))/2
    print('Accuracy of %5s : %2d %%' % (classes[task[0]], 100 * class_correct[0] / class_total[0]))
    print('Accuracy of %5s : %2d %%' % (classes[task[1]], 100 * class_correct[1] / class_total[1]))
    print('Avarage accuracy of %5s : %2d %%' % (task, 100* avarage_accuracy))
    return avarage_accuracy


def main():
    # Use a standard transformation on all datasets.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Load the CIFAR10 training and test sets.
    train_set = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    # Create task datasets and data loaders for both training and testing.
    tasks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

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

    net.train()
    initial_accuracy = []
    i = 0
    for task in tasks:
        print(f'Network with task', i, 'with class ', classes[task[0]], classes[task[1]])
        net.add_task(len(task))
        net.to(device)
        net.set_tasks([i])
        net.train()
        train(net, task_train_dls[i], task, criterion, optimizer, num_epochs)
        initial_accuracy.append(task_accuracy(net, task_test_dls[i], task))
        i = i + 1

    for i in range(len(initial_accuracy)):
        writer.add_scalar("accuracy", initial_accuracy[i], i+1)

    writer.close()
    net.eval()
    final_accuracy = []
    i = 0
    for task in tasks:
        final_accuracy.append(task_accuracy(net, task_test_dls[i], task))
        i = i + 1
    for i in range(len(final_accuracy)):
         writer.add_scalar("accuracy", final_accuracy[i], i+1)

    x = [1, 2, 3, 4, 5]

    plt.plot(x, initial_accuracy, color="red", marker="o", label="initial")
    plt.plot(x, final_accuracy, color="blue", marker="o", label="final")
    plt.legend()
    plt.show()

    forgetting = []


    for i in range(5):
        forgetting.append( initial_accuracy[i] - final_accuracy[i])
        writer.add_scalar("forgetting", forgetting[i], i+1)
    plt.plot(x, forgetting, label="forgetting")
    plt.legend()
    plt.show()





if __name__ == '__main__':
    main()
