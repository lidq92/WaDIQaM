# Implemented by Dingquan Li
# Email: dingquanli@pku.edu.cn
# Date: 2018/4/18

import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
import random
import h5py


def default_loader(path):
    return Image.open(path).convert('RGB')


def RandomCropPatches(im, ref, model_type='FR', th=32, tw=32, n_patches=32):
    w, h = im.size
    crops = ()
    ref_crops = ()
    for k in range(n_patches):
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        crops = (to_tensor(im.crop((j, i, j + tw, i + th))),) + crops
        if model_type=='FR': # 
            ref_crop = to_tensor(ref.crop((j, i, j + tw, i + th)))
            ref_crops = (ref_crop,) + ref_crops
    return (torch.stack(crops),torch.stack(ref_crops))


def FullyNonoverlappingCropPatches(im, ref, model_type='FR', 
                                   th=32, tw=32):
    w, h = im.size
    crops = ()
    ref_crops = ()
    for i in range(0, h - th, th):
        for j in range(0, w - tw, tw):
            crops = (to_tensor(im.crop((j, i, j + tw, i + th))),) + crops
            if model_type=='FR':
                ref_crop = to_tensor(ref.crop((j, i, j + tw, i + th)))
                ref_crops = (ref_crop,) + ref_crops
    return (torch.stack(crops),torch.stack(ref_crops))


class IQADataset(Dataset):
    def __init__(self, conf, EXP_ID, status='train', loader=default_loader):
        self.loader = loader
        self.im_dir = conf['im_dir']
        self.ref_dir = conf['ref_dir']
        self.status =status
        self.type = conf['type']
        self.th = conf['th']
        self.tw = conf['tw']
        self.n_patches = conf['n_patches']
        datainfo = conf['datainfo']

        Info = h5py.File(datainfo)
        index = Info['index'][:, int(EXP_ID) % 1000] # 
        ref_ids = Info['ref_ids'][0, :] #
        test_ratio = conf['test_ratio']  #
        train_ratio = conf['train_ratio']
        trainindex = index[:int(train_ratio * len(index))]
        testindex = index[int((1-test_ratio) * len(index)):]
        train_index, val_index, test_index = [],[],[]
        for i in range(len(ref_ids)):
            train_index.append(i) if (ref_ids[i] in trainindex) else \
                test_index.append(i) if (ref_ids[i] in testindex) else \
                    val_index.append(i)
        if status == 'train':
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
        if status == 'test':
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
        if status == 'val':
            self.index = val_index
            print("# Val Images: {}".format(len(self.index)))
        print('Index:')
        print(self.index)

        self.label = Info['subjective_scores'][0, self.index] #
        self.label_std = Info['subjective_scoresSTD'][0, self.index] #
        self.im_names = [Info[Info['im_names'][0, :][i]].value.tobytes()\
                        [::2].decode() for i in self.index]
        self.ref_names = [Info[Info['ref_names'][0, :][i]].value.tobytes()\
                         [::2].decode() for i in (ref_ids[self.index]-1).astype(int)]
        
        if self.status != 'train':
            self.data = []
            for idx in range(len(self.index)):
                im = self.loader(os.path.join(self.im_dir, 
                                        self.im_names[idx]))
                ref = self.loader(os.path.join(self.ref_dir, 
                                        self.ref_names[idx]))
                self.data.append(RandomCropPatches(im, ref, self.type,
                                        self.th, self.tw, self.n_patches))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if self.status == 'train':
            im = self.loader(os.path.join(self.im_dir, self.im_names[idx]))
            ref = self.loader(os.path.join(self.ref_dir, self.ref_names[idx]))
            crops, ref_crops = RandomCropPatches(im, ref, self.type, 
                                        self.th, self.tw, self.n_patches)
        else:
            crops, ref_crops = self.data[idx]
        return (crops, ref_crops, torch.Tensor([self.label[idx],]), 
                torch.Tensor([self.label_std[idx],]))
