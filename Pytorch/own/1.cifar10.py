#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-06-03 15:00:32
# Last Modified by: xiaofeng
# Last Modified time: 2018-06-03 15:00:32
from __future__ import print_function
import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
show = ToPILImage()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])
trainset = tv.datasets.CIFAR10(
    root='/Users/xiaofeng/Code/Github/dataset/test_datasets/cifar/', train=True, download=True,
    transform=transform)
trainloader = t.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=1)
testset = tv.datasets.CIFAR10(
    root='/Users/xiaofeng/Code/Github/dataset/test_datasets/cifar/', train=False, download=True,
    transform=transform)
testloader = t.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=1)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

(data, label) = trainset[50]
print(classes[label])

# (data + 1) / 2是为了还原被归一化的数据
show((data + 1) / 2).resize((100, 100))

# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# print(' '.join('%11s' % classes[labels[j]] for j in range(8)))
# show(tv.utils.make_grid((images + 1) / 2)).resize((800, 100))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)


criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
t.set_num_threads(8)
for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        # 输入数据
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印log信息
        running_loss += loss.data[0]
        if i % 2000 == 1999:  # 每2000个batch打印一下训练状态
            print('[%d, %5d] loss: %.3f'
                  % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')
