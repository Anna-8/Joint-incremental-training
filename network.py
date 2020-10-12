import torch
import torch.nn.functional as F
import torch.nn as nn

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

