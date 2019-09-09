# Implemented by Dingquan Li
# Email: dingquanli@pku.edu.cn
# Date: 2019/3/5


import h5py, os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


def default_loader(path, channel=3):
    if channel == 1:
        return Image.open(path).convert('L')
    else:
        assert (channel == 3)
        return Image.open(path).convert('RGB')  #


def RandomCropPatches(im, ref=None, patch_size=32, n_patches=32):
    """
    Random Crop Patches
    :param im: the distorted image
    :param ref: the reference image if FR-IQA is considered (default: None)
    :param patch_size: patch size (default: 32)
    :param n_patches: numbers of patches (default: 32)
    :return: patches
    """
    w, h = im.size

    patches = ()
    ref_patches = ()
    for i in range(n_patches):
        w1 = np.random.randint(low=0, high=w-patch_size+1)
        h1 = np.random.randint(low=0, high=h-patch_size+1)
        patch = to_tensor(im.crop((w1, h1, w1 + patch_size, h1 + patch_size)))
        patches = patches + (patch,)
        if ref is not None:
            ref_patch = to_tensor(ref.crop((w1, h1, w1 + patch_size, h1 + patch_size)))
            ref_patches = ref_patches + (ref_patch,)

    if ref is not None:
        return torch.stack(patches), torch.stack(ref_patches)
    else:
        return torch.stack(patches)


def NonOverlappingCropPatches(im, ref=None, patch_size=32):
    """
    NonOverlapping Crop Patches
    :param im: the distorted image
    :param ref: the reference image if FR-IQA is considered (default: None)
    :param patch_size: patch size (default: 32)
    :return: patches
    """
    w, h = im.size

    patches = ()
    ref_patches = ()
    stride = patch_size
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patches = patches + (patch,)
            if ref is not None:
                ref_patch = to_tensor(ref.crop((j, i, j + patch_size, i + patch_size)))
                ref_patches = ref_patches + (ref_patch,)

    if ref is not None:
        return torch.stack(patches), torch.stack(ref_patches)
    else:
        return torch.stack(patches)


class IQADataset(Dataset):
    """
    IQA Dataset
    """
    def __init__(self, args, status='train', loader=default_loader, n_splits=1000):
        """
        :param args:
        :param status: train/val/test
        :param loader: image loader
        :param n_splits: number of train/val/test splits that you have prepared (default: 1000)
        """
        self.status = status
        self.patch_size = args.patch_size
        self.n_patches = args.n_patches

        Info = h5py.File(args.data_info)
        index = Info['index'][:, args.exp_id % n_splits]  #
        ref_ids = Info['ref_ids'][0, :]  #
        trainindex = index[:int(args.train_ratio * len(index))]
        testindex = index[int((1 - args.test_ratio) * len(index)):]
        train_index, val_index, test_index = [], [], []
        for i in range(len(ref_ids)):
            train_index.append(i) if (ref_ids[i] in trainindex) else \
                test_index.append(i) if (ref_ids[i] in testindex) else \
                    val_index.append(i)
        if 'train' in status:
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
        if 'test' in status:
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
        if 'val' in status:
            self.index = val_index
            print("# Val Images: {}".format(len(self.index)))
        print('Index:')
        print(self.index)

        self.mos = Info['subjective_scores'][0, self.index]  #
        self.mos_std = Info['subjective_scoresSTD'][0, self.index]  #
        im_names = [Info[Info['im_names'][0, :][i]].value.tobytes()[::2].decode() for i in self.index]
        ref_names = [Info[Info['ref_names'][0, :][i]].value.tobytes()[::2].decode()
                     for i in (ref_ids[self.index]-1).astype(int)]

        self.patches = ()
        self.label = []
        self.label_std = []
        self.ims = []
        self.refs = []
        for idx in range(len(self.index)):
            # print("Preprocessing Image: {}".format(im_names[idx]))
            im = loader(os.path.join(args.im_dir, im_names[idx]))
            if args.ref_dir is None or args.model == 'WaDIQaM-NR' or args.model == 'DIQaM-NR':
                ref = None
            else:
                ref = loader(os.path.join(args.ref_dir, ref_names[idx]))

            self.label.append(self.mos[idx])
            self.label_std.append(self.mos_std[idx])

            if status == 'train':
                self.ims.append(im)
                self.refs.append(ref)
            elif status == 'test' or status == 'val':
                patches = NonOverlappingCropPatches(im, ref, args.patch_size)  # Random or Non Overlapping Crop?
                self.patches = self.patches + (patches,)  #

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if self.status == 'train':
            patches = RandomCropPatches(self.ims[idx], self.refs[idx], self.patch_size, self.n_patches)
        else:
            patches = self.patches[idx]
        return patches, (torch.Tensor([self.label[idx], ]), torch.Tensor([self.label_std[idx], ]))
