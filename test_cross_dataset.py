"""
Test Cross Dataset
    For help
    ```bash
    python test_cross_dataset.py --help
    ```
 Date: 2018/5/26
"""

from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from IQADataset import RandomCropPatches
from IQAmodel import FRnet, NRnet
import numpy as np
import h5py, os


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
        x, x_ref = data

        if self.use_cuda:
            x = x.cuda()
            x_ref = x_ref.cuda()
        x = Variable(x, volatile=not train)
        x_ref = Variable(x_ref, volatile=not train)     


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
        elif self.top == "patchwise":
            a = Variable(torch.ones_like(h.data), volatile=not train)
        # print(h.size())
        # print(a.size())
        q = torch.sum(h * a) / torch.sum(a)
        return q


if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch WaDIQaM-FR test on the whole cross dataset')
    parser.add_argument("--dist_dir", type=str, default=None,
                        help="distorted images dir.")
    parser.add_argument("--ref_dir", type=str, default=None,
                        help="reference images dir.")
    parser.add_argument("--names_info", type=str, default=None,
                        help=".mat file that includes image names in the dataset.")
    parser.add_argument("--model_file", type=str, default='models/WaDIQaM-FR-TID2008',
                        help="model file (default: WaDIQaM-FR)")
    parser.add_argument("--save_path", type=str, default='scores',
                        help="save path (default: score)")

    args = parser.parse_args()

    model = FRnet(top="weighted", use_cuda=torch.cuda.is_available())

    model.load_state_dict(torch.load(args.model_file))
    if torch.cuda.is_available():
        model = model.cuda()

    Info = h5py.File(args.names_info)
    im_names = [Info[Info['im_names'][0, :][i]].value.tobytes()\
                        [::2].decode() for i in range(len(Info['im_names'][0, :]))]
    ref_names = [Info[Info['ref_names'][0, :][i]].value.tobytes()\
                        [::2].decode() for i in range(len(Info['ref_names'][0, :]))]

    model.eval()
    scores = []   
    for i in range(len(im_names)):
        im = Image.open(os.path.join(args.dist_dir, im_names[i])).convert('RGB')
        ref = Image.open(os.path.join(args.ref_dir, ref_names[i])).convert('RGB')
        data = RandomCropPatches(im, ref)
        score = model(data, train=False)
        if torch.cuda.is_available():
            score = score.data.cpu().numpy()
        else:
            score = score.data.numpy()
        print(score[0])
        scores.append(score[0])
    np.save(args.save_path, scores)
