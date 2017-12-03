import os
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import math
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import pylab
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from Vgg import *

torch.manual_seed(0)
torch.cuda.manual_seed(0)

batch_size = 128
learning_rate = 0.01
learning_rate_decay = 0.0005
momentum = 0.9
epoch_step = 25
max_epoch = 300

transform_train = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)


save_dir = "./save"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2./n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


# net = Vgg()
# net = VggDropBN()
# net = VggDropout()
net = VggBN()
net.apply(weights_init)
net.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=learning_rate_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=epoch_step, gamma=0.5)

test_accuracies = np.zeros(max_epoch)
for epoch in range(max_epoch):  # loop over the dataset multiple times
    pbar = tqdm(trainloader)
    pbar.mininterval = 1 # update the processing bar at least 1 second

    """
        Initial Check
    """
    net.eval()
    
    if epoch == 0:
        print('\033[0;31mInitial Check: \033[0m')
        running_loss, correct, total = 0., 0., 0.
        for i, data in enumerate(testloader, 0):
            images, labels = data
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss = running_loss * (i/(i+1.)) + loss.data[0] * (1./(i+1.) )
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        print('Loss on the test images: %f  ......  should be 2.3' % running_loss)
        print('Accuracy on the test images: %f %% ......  should be 10%%' % (100. * correct / total))

    """
        Training ...
    """
    net.train()

    running_loss, correct, total = 0., 0., 0.
    scheduler.step()

    for i, data in enumerate(pbar, 0):
        # get the inputs
        inputs, labels = data
        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # update statistics
        running_loss = running_loss * (i/(i+1.)) + loss.data[0] * (1./(i+1.) )
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
    print('\033[0;32m Statistics on epoch :%d learning rate: %f\033[0m' %(epoch, scheduler.get_lr()[0]))
    print('Train Loss : %f Train Accuracy: %f %%' % (running_loss, 100. * correct / total))

    """
        Testing ...
    """
    net.eval()

    correct, total = 0., 0.
    for data in testloader:
        images, labels = data
        images = Variable(images.cuda())
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()

    print('Test Accuracy: \033[1;33m%f %%\033[0m' % (100. * correct / total))
    test_accuracies[epoch] = 100. * correct / total

    """
        Saving model and accuracies, and ploting
    """
    np.save('./save/accuracies.npy', test_accuracies)
    torch.save(net.state_dict(), './save/model.%d.pkl' %epoch)

    plt.figure()
    pylab.xlim(0, max_epoch + 1)
    pylab.ylim(0, 100)
    plt.plot(range(1, max_epoch +1), test_accuracies)
    plt.savefig('./save/accuracies.png')
    plt.close()


