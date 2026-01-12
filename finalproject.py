########################
#EEE419 Final Project
#Caitlyn Blythe
########################

import numpy as np
import torch
import torch.nn
import torch.optim
import torch.nn.functional as F
import time
from torchvision import datasets
from torchvision import transforms

BATCH_SIZE = 64
NUM_CLASSES = 2
EPOCHS = 3
CLASS_IDX = [3,5]    #indices of cats/dogs

#create class that filters the cifar10 to only include cats and dogs
class cifar10_filter(datasets.CIFAR10):
    def __init__(self, root, train, transform):
        super().__init__(root=root,train=train,transform=transform)
        filt = np.isin(self.targets,CLASS_IDX)
        self.data = self.data[filt]
        self.targets = np.array(self.targets)[filt]
        self.targets = [0 if t == CLASS_IDX[0] else 1 for t in self.targets]    #make cats and dogs indices 0 and 1

#create class for rgb image network (3 input channels)
class cifar10_rgb_net(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3),padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=1)
        self.mpool = torch.nn.MaxPool2d(kernel_size=2)
        self.drop1 = torch.nn.Dropout(p=0.25)
        self.flat = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(in_features=64*16*16,out_features=128)
        self.drop2 = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(in_features=128,out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.mpool(x)
        x = self.drop1(x)
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x
#create class for grayscale image network (1 input channel)
class cifar10_grayscale_net(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(3,3),padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=1)
        self.mpool = torch.nn.MaxPool2d(kernel_size=2)
        self.drop1 = torch.nn.Dropout(p=0.25)
        self.flat = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(in_features=64*16*16,out_features=128)
        self.drop2 = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(in_features=128,out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.mpool(x)
        x = self.drop1(x)
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x

#function to train the network, loop over epochs
def train(net,train_loader,criterion,optimizer):
    net.train()
    for epoch in range(EPOCHS):
        for batch_idx, (images,labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                batch_loss = loss.mean().item

#function to evaluate the accuracy of the networks
def performance(net,test_loader):
    num_correct = 0
    num_attempts = 0
    for images, labels in test_loader:
        with torch.no_grad():
            outputs = net(images)
            guesses = torch.argmax(outputs,1)
            num_guess = len(guesses)
            num_right = torch.sum(labels == guesses).item()
            num_correct += num_right
            num_attempts += num_guess
    return num_correct/num_attempts

#transform the datasets (for grayscale, we need to convert first using transforms.Grayscale
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
transform = transforms.Compose([to_tensor, normalize])
gray_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor(),transforms.Normalize([0.5,],[0.5,])])

#load train and test data for the rgb network
rgb_trainset = cifar10_filter('~/CIFAR10_data/train',train=True,transform=transform)
rgb_train_loader = torch.utils.data.DataLoader(rgb_trainset, batch_size=BATCH_SIZE, shuffle=True)
rgb_testset = cifar10_filter('~/CIFAR10_data/test',train=False,transform=transform)
rgb_test_loader = torch.utils.data.DataLoader(rgb_testset, batch_size=BATCH_SIZE, shuffle=True)

#initialize the object to prep for training
rgb_net = cifar10_rgb_net(NUM_CLASSES)
rgb_criterion = torch.nn.CrossEntropyLoss()
rgb_optimizer = torch.optim.Adadelta(rgb_net.parameters())

#load train and test data for grayscale network
gray_trainset = cifar10_filter('~/CIFAR10_data/train',train=True,transform=gray_transform)
gray_train_loader = torch.utils.data.DataLoader(gray_trainset, batch_size=BATCH_SIZE, shuffle=True)
gray_testset = cifar10_filter('~/CIFAR10_data/train',train=False,transform=gray_transform)
gray_test_loader = torch.utils.data.DataLoader(gray_testset, batch_size=BATCH_SIZE, shuffle=True)

#initialize and prep for training
gray_net = cifar10_grayscale_net(NUM_CLASSES)
gray_criterion = torch.nn.CrossEntropyLoss()
gray_optimizer = torch.optim.Adadelta(gray_net.parameters())

#train the rgb network and get results for accuracy and runtime
start_rgb_time = time.time()
train(rgb_net,rgb_train_loader,rgb_criterion,rgb_optimizer)
end_rgb_time = time.time()
rgb_accuracy = performance(rgb_net,rgb_test_loader)
rgb_accuracy = rgb_accuracy * 100
rgb_runtime = end_rgb_time - start_rgb_time

#train the grayscale network and get results for accuracy and runtime
start_gray_time = time.time()
train(gray_net,gray_train_loader,gray_criterion,gray_optimizer)
end_gray_time = time.time()
gray_accuracy = performance(gray_net,gray_test_loader)
gray_accuracy = gray_accuracy * 100
gray_runtime = end_gray_time - start_gray_time

#print results and recommended algorithm
print(f'RGB Accuracy: {rgb_accuracy:.2f}%')
print(f'Grayscale Accuracy: {gray_accuracy:.2f}%')
print(f'RGB Runtime: {rgb_runtime:.5f}s seconds')
print(f'Grayscale Runtime: {gray_runtime:.5f}s seconds')
print('Caitlyn Blythe recommends RGB algorithm')
