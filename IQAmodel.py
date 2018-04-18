# Implemented by Dingquan Li
# Email: dingquanli@pku.edu.cn
# Date: 2018/4/18
#
# TODO: hyper-parameters to config
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class FRnet(nn.Module):
    def __init__(self, top="patchwise", use_cuda=True):
        super(FRnet, self).__init__()
        self.conv1  = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2  = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3  = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4  = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5  = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6  = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7  = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8  = nn.Conv2d(256, 256, 3, padding=1)
        self.conv9  = nn.Conv2d(256, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.fc1    = nn.Linear(512*3, 512)
        self.fc2    = nn.Linear(512, 1)
        self.fc1_a  = nn.Linear(512*3, 512)
        self.fc2_a  = nn.Linear(512, 1)
        self.top = top
        self.use_cuda = use_cuda

    def extract_features(self, x):

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 2)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pool2d(h, 2)

        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.max_pool2d(h, 2)

        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.max_pool2d(h, 2)

        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.max_pool2d(h, 2)

        h = h.view(-1,512)
        return h

    def forward(self, data, train=True):
        x, x_ref, y, y_std = data

        x = x.view(-1, 3, 32, 32) #
        x_ref = x_ref.view(-1, 3, 32, 32) #

        self.n_images = y.shape[0]
        self.n_patches = x.shape[0]
        self.n_patches_per_image = self.n_patches // self.n_images
        if self.use_cuda:
            x = x.cuda()
            x_ref = x_ref.cuda()
            y = y.cuda()
        x = Variable(x, volatile=not train)
        x_ref = Variable(x_ref, volatile=not train)
        y = Variable(y)            


        h = self.extract_features(x)
        h_ref = self.extract_features(x_ref)
        h = torch.cat((h - h_ref, h, h_ref), 1)

        h_ = h # save intermediate features
        self.h = h_

        h = F.dropout(F.relu(self.fc1(h_)), p=0.5, training=train)
        h = self.fc2(h)

        if self.top == "weighted":
            a = F.dropout(F.relu(self.fc1_a(h_)), p=0.5, training=train)
            a = F.relu(self.fc2_a(a)) + 0.000001 # small constant
            t = y
            self.weighted_loss(h, a, t)
        elif self.top == "patchwise":
            a = Variable(torch.ones_like(h.data), volatile=not train)
            t = y.repeat(self.n_patches_per_image,1)
            self.patchwise_loss(h, a, t)

        if train:
            return self.loss
        else:
            self.q = Variable(torch.ones_like(y.data), volatile=not train)
            for i in range(self.n_images):
                self.q[i] = torch.sum(self.y[i] * self.a[i]) / torch.sum(self.a[i])
            return self.loss, self.y, self.a, self.q

    def patchwise_loss(self, h, a, t):
        self.loss = torch.sum(torch.abs(h - t.view(-1, 1)))
        self.loss /= self.n_patches
        if self.n_images > 1:
            h = torch.split(h, self.n_patches_per_image, 0)
            a = torch.split(a, self.n_patches_per_image, 0)
        else:
            h, a = [h], [a]
        self.y = h
        self.a = a

    def weighted_loss(self, h, a, t):
        self.loss = 0
        if self.n_images > 1:
            h = torch.split(h, self.n_patches_per_image, 0)
            a = torch.split(a, self.n_patches_per_image, 0)
        else:
            h, a, t = [h], [a], [t]

        for i in range(self.n_images):
            y = torch.sum(h[i] * a[i], 0) / torch.sum(a[i], 0)
            self.loss += torch.abs(y - t[i])
        self.loss /= self.n_images
        self.y = h
        self.a = a

class NRnet(nn.Module):
    def __init__(self, top="patchwise", use_cuda=True):
        super(NRnet, self).__init__()
        self.conv1  = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2  = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3  = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4  = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5  = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6  = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7  = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8  = nn.Conv2d(256, 256, 3, padding=1)
        self.conv9  = nn.Conv2d(256, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.fc1    = nn.Linear(512, 512)
        self.fc2    = nn.Linear(512, 1)
        self.fc1_a  = nn.Linear(512, 512)
        self.fc2_a  = nn.Linear(512, 1)
        self.top = top
        self.use_cuda = use_cuda

    def forward(self, data, train=True):
        x, x_ref, y, y_std = data
        x = x.view(-1, 3, 32, 32) #

        self.n_images = y.shape[0]
        self.n_patches = x.shape[0]
        self.n_patches_per_image = self.n_patches // self.n_images
        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()
        x = Variable(x, volatile=not train)
        y = Variable(y)


        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 2)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pool2d(h, 2)

        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.max_pool2d(h, 2)

        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.max_pool2d(h, 2)

        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.max_pool2d(h, 2)

        h = h.view(-1,512)

        h_ = h
        self.h = h_

        h = F.dropout(F.relu(self.fc1(h_)), p=0.5, training=train)
        h = self.fc2(h)

        if self.top == "weighted":
            a = F.dropout(F.relu(self.fc1_a(h_)), p=0.5, training=train)
            a = F.relu(self.fc2_a(a)) + 0.000001 # small constant
            t = y
            self.weighted_loss(h, a, t)
        elif self.top == "patchwise":
            a = Variable(torch.ones_like(h.data), volatile=not train)
            t = y.repeat(n_patches,1)
            self.patchwise_loss(h, a, t)

        if train:
            return self.loss
        else:
            self.q = Variable(torch.ones_like(y.data), volatile=not train)
            for i in range(self.n_images):
                self.q[i] = torch.sum(self.y[i] * self.a[i]) / torch.sum(self.a[i])
            return self.loss, self.y, self.a, self.q

    def patchwise_loss(self, h, a, t):
        self.loss = torch.sum(torch.abs(h - t.view(-1, 1)))
        self.loss /= self.n_patches
        if self.n_images > 1:
            h = torch.split(h, self.n_patches_per_image, 0)
            a = torch.split(a, self.n_patches_per_image, 0)
        else:
            h, a = [h], [a]
        self.y = h
        self.a = a

    def weighted_loss(self, h, a, t):
        self.loss = 0
        if self.n_images > 1:
            h = torch.split(h, self.n_patches_per_image, 0)
            a = torch.split(a, self.n_patches_per_image, 0)
        else:
            h, a, t = [h], [a], [t]

        for i in range(self.n_images):
            y = torch.sum(h[i] * a[i], 0) / torch.sum(a[i], 0)
            self.loss += torch.abs(y - t[i])
        self.loss /= self.n_images
        self.y = h
        self.a = a