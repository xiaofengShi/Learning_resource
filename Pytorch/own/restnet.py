#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-06-08 19:04:48
# Last Modified by: xiaofeng
# Last Modified time: 2018-06-08 19:04:48

import os
import sys

import torch as t
from PIL import Image
from torch import nn
from torch.autograd import Variable as V
from torchvision.transforms import ToPILImage, ToTensor
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
to_tensor = ToTensor()
to_pil = ToPILImage
lena = Image.open('./chapter4-神经网络工具箱nn/imgs/lena.png')

# lena.show()

input = to_tensor(lena).unsqueeze(0)
# 锐化卷积核
kernel = t.ones(3, 3) / -9.
kernel[1][1] = 1
conv = nn.Conv2d(1, 1, (3, 3), 1, bias=False)
conv.weight.data = kernel.view(1, 1, 3, 3)
out = conv(V(input))
to_pil(out.data.squeeze(0))
pool = nn.AvgPool2d(2, 2)
print(list(pool.parameters()))
out = pool(V(input))
to_pil(out.data.squeeze(0))

plt.show()
