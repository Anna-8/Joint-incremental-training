
from itertools import islice
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import SGD
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter


# CLASSE IL DATASET FILTRATO
class FilteredDataset(Dataset):
    def __init__(self, dataset, indices, offset=0):
        self.indices = indices # indici dei target di cui ci interessa il target
        self.original_indices = [i for i in range(len(dataset.targets)) if dataset.targets[i] in indices] # filtraggio degli indici del dataset che voglio nel subset
        self.dataset = Subset(dataset, self.original_indices)
        self.original2task = { indices[i]  : offset + i for i in range(0,  len(indices) ) } # remapping degli indici con chiavi i valori degli indici del del task
        self.task2original = dict({ (val, key) for (key, val) in self.original2task.items() }) # remapping degli indici con chiavi i valori degli indici del del task

    def __getitem__(self, idx):
        (x, y) = self.dataset[idx]
        return (x, self.original2task[y])

    def __len__(self):
        return len(self.original_indices)

# A CNN for multi-task classification.
class Net(nn.Module):
    # Constructor takes in input the important hyper parameters of the network.
    def __init__(self, image_size=32, image_channels=3, kernel=3, numFM1=32, numFM2=64, numFC1=256, numFC2=256):
        super(Net, self).__init__()
        self.numFC2 = numFC2
        self.conv1 = nn.Conv2d(image_channels, numFM1, kernel, padding=kernel//2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(numFM1, numFM2, kernel, padding=kernel//2)
        self.fc1 = nn.Linear(numFM2 * ((image_size // 4)**2), numFC1)
        self.fc2 = nn.Linear(numFC1, numFC2)
        self.task_fcs = []       # Will hold all Linear layers for classification heads.
        self.current_tasks = []  # Selects which task(s) are currently active.

    # Forward pass of the network: returns ONLY currently selected tasks.
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Concatenates the classification heads of selected tasks.
        outputs = torch.cat([F.log_softmax(self.task_fcs[t](x)) for t in self.current_tasks], 1)
        return outputs

    # Add a new classification head with num_classes outputs.
    def add_task(self, num_classes):
        fc = nn.Linear(self.numFC2, num_classes)
        self.add_module(name=f'task{len(self.task_fcs)}_fc', module=fc)
        self.task_fcs.append(fc)

    # Set the current task(s) -- takes a *LIST* of task ids.
    def set_tasks(self, tasks):
        self.current_tasks = tasks

# Function to visualize a batch of images from a dataloader over a filtered
# dataset. This shows each image, the task label within each task, and the
# original label. This lets us verify that FilteredDataset is working.
def verify_dataloaders(dls):
    for (t, dl) in enumerate(dls):
        plt.figure()
        plt.title(f'Task: {t}')
        batch = dl.__iter__().__next__()
        for (i, (im, label)) in islice(enumerate(zip(*batch)), 9):
            plt.subplot(3, 3, i+1)
            plt.axis(False)
            plt.imshow(im[0])
            plt.title(f'TL: {label}, OL: {dl.dataset.task2original[int(label)]}')
        plt.show()

# Function that verifies that the network is working the way we need it to.
# Verifies that only selected task heads are returned, and that when training
# only currently selected heads are modified.
def verify_network(net, dls):
    # Get a batch of images and labels from both dataloader.
    [(xs0, ys0)] = list(islice(task_train_dls[0], 1))  # from the first data loader take the first group of images
    [(xs1, ys1)] = list(islice(task_train_dls[1], 1))  # from the second data loader

    # Add classification heads for all tasks to the network.
    net.add_task(5)
    net.add_task(5)

    # Run images through the network with various tasks set.
    net.set_tasks([0])
    print(f'Network output with task 0 set:\n{net(xs0)}\n')
    net.set_tasks([1])
    print(f'Network output with task 1 set:\n{net(xs0)}\n')
    net.set_tasks([0, 1])
    print(f'Network output with tasks 0 and 1 set:\n{net(xs0)}')

    # Print the norm of the fc weights for tracking -- this is a cheap way to
    # see if the weights are being changed during training.
    print(f'Task 0 weight norm BEFORE training: {torch.norm(net.task_fcs[0].weight)}')
    print(f'Task 1 weight norm BEFORE training: {torch.norm(net.task_fcs[1].weight)}')

    # Run a single optimization loop on task 0.
    optimizer = SGD(net.parameters(), lr=1e-3)
    net.set_tasks([0])
    net.train()
    # sets the module in training mode
    # has any effect only on certain modules like dropout, which behave differently in training/evaluation mode
    for (i, (xs, ys)) in enumerate(dls[0]):
        # i is the position and (xs, ys) the image and the label
        optimizer.zero_grad()
        output = net(xs)
        loss = F.nll_loss(output, ys)
        loss.backward()
        optimizer.step()

    # Print the norm of the fc weights for tracking.
    print(f'Task 0 weight norm AFTER training task 0: {torch.norm(net.task_fcs[0].weight)}')
    print(f'Task 1 weight norm AFTER training task 0: {torch.norm(net.task_fcs[1].weight)}')

    # And do the same thing for task 1.
    optimizer.zero_grad()
    net.set_tasks([1])
    for (i, (xs, ys)) in enumerate(dls[1]):
        optimizer.zero_grad()
        output = net(xs)
        loss = F.nll_loss(output, ys)
        loss.backward()
        optimizer.step()

    # Print the norm of the fc weights for tracking.
    print(f'Task 0 weight norm AFTER training task 1: {torch.norm(net.task_fcs[0].weight)}')
    print(f'Task 1 weight norm AFTER training task 1: {torch.norm(net.task_fcs[1].weight)}')

# Put all experiment hyperparameters HERE.
batch_size = 16
verify_dls = False
verify_net = True

if __name__ == '__main__':
    writer = SummaryWriter('runs/experiment_1')

    # Use a standard transformation on ALL datasets.
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

    # Load the MNIST training and test sets.
    ds_train = MNIST('./data/', train=True, download=True, transform=transform)
    ds_test = MNIST('./data/', train=False, download=True, transform=transform)

    # Create our task datasets and dataloaders for both training and testing.
    tasks = [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9)]
    # define the datasets for training
    task_train_dss = [FilteredDataset(ds_train, classes) for classes in tasks]
    task_train_dls = [DataLoader(ds, batch_size=batch_size) for ds in task_train_dss]
    # define the datasets for validation
    task_test_dss = [FilteredDataset(ds_test, classes) for classes in tasks]
    # define the data loaders for validation
    task_test_dls = [DataLoader(ds, batch_size=batch_size) for ds in task_test_dss]

    # If verify flag is set, run the dataloader verification.
    if verify_dls:
        verify_dataloaders(task_test_dls)

    # Instantiate a network.
    net = Net()

    #If verify flag set, run network verification.
    if verify_net:
        verify_network(net, task_train_dls)


def netAccuracy(net, val_loader):
    correct = 0
    total = 0
    # not update parameters
    with torch.no_grad():
        for data in val_loader:
            # images and labels contain a number of training examples depends on the batch size
            images, labels = data

            outputs = net(images)
            # print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)  # labels.size returns [N,1] where N is the batch size
            correct += (predicted == labels).sum().item()
            # item to cast to a Python int an integer tensor
            # sum gives the number of items in the batch where the prediction and ground truth agree.

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

def classAccuracy(net, val_loader):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()  # squeeze elimina le dimensioni uguali a 1
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % ( classes[i], 100 * class_correct[i] / class_total[i]))