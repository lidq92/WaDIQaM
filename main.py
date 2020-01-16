"""
PyTorch 1.1 implementation of the following paper:
Bosse S, Maniry D, Müller K R, et al. Deep neural networks for no-reference and full-reference image quality assessment.
IEEE Transactions on Image Processing, 2018, 27(1): 206-219.

 Requirements: See requirements.txt.
    ```bash
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

 Usage:
    Start tensorboard:
    ```bash
    tensorboard --logdir=tensorboard_logs --port=6006
    ```
    Run the main.py:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python main.py --exp_id=0
    ```
    for more arguments, execute `python main.py --help`

 Implemented by Dingquan Li
 Email: dingquanli@pku.edu.cn
 Date: 2019/9/9
"""

from argparse import ArgumentParser

import os
import numpy as np
import random
from scipy import stats
import h5py
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics.metric import Metric

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


def default_loader(path, channel=3):
    """
    :param path: image path
    :param channel: # image channel
    :return: image
    """
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


class IQADataset_less_memory(Dataset):
    """
    IQA Dataset (less memory)
    """
    def __init__(self, args, status='train', loader=default_loader):
        """
        :param args:
        :param status: train/val/test
        :param loader: image loader
        """
        self.status = status
        self.patch_size = args.patch_size
        self.n_patches = args.n_patches
        self.loader = loader

        Info = h5py.File(args.data_info, 'r')
        index = Info['index']
        index = index[:, args.exp_id % index.shape[1]]
        ref_ids = Info['ref_ids'][0, :]  #

        K = args.K_fold
        k = args.k_test
        testindex = index[int((k-1)/K * len(index)):int(k/K * len(index))]
        valindex = index[range(-int((5-k)/K * len(index)), -int((4-k)/K * len(index)))]
        train_index, val_index, test_index = [], [], []
        for i in range(len(ref_ids)):
            if ref_ids[i] in testindex:
                test_index.append(i)
            elif ref_ids[i] in valindex:
                val_index.append(i)
                train_index.append(i)  #
            else:
                train_index.append(i)
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

        self.scale = Info['subjective_scores'][0, :].max()
        self.mos = Info['subjective_scores'][0, self.index] / self.scale #
        self.mos_std = Info['subjective_scoresSTD'][0, self.index] / self.scale #
        im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()[::2].decode() for i in self.index]
        ref_names = [Info[Info['ref_names'][0, :][i]][()].tobytes()[::2].decode()
                     for i in (ref_ids[self.index]-1).astype(int)]

        self.patches = ()
        self.label = []
        self.label_std = []
        self.im_names = []
        self.ref_names = []
        for idx in range(len(self.index)):
            self.im_names.append(os.path.join(args.im_dir, im_names[idx]))
            if args.ref_dir is None or 'NR' in args.model:
                self.ref_names.append(None)
            else:
                self.ref_names.append(os.path.join(args.ref_dir, ref_names[idx]))

            self.label.append(self.mos[idx])
            self.label_std.append(self.mos_std[idx])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        im = self.loader(self.im_names[idx])
        if self.ref_names[idx] is None:
            ref = None
        else:
            ref = self.loader(self.ref_names[idx])

        if self.status == 'train':
            patches = RandomCropPatches(im, ref, self.patch_size, self.n_patches)
        else:
            patches = NonOverlappingCropPatches(im, ref, self.patch_size)
        return patches, (torch.Tensor([self.label[idx], ]), torch.Tensor([self.label_std[idx], ]))


class IQADataset(Dataset):
    """
    IQA Dataset
    """
    def __init__(self, args, status='train', loader=default_loader):
        """
        :param args:
        :param status: train/val/test
        :param loader: image loader
        """
        self.status = status
        self.patch_size = args.patch_size
        self.n_patches = args.n_patches

        Info = h5py.File(args.data_info, 'r')
        index = Info['index']
        index = index[:, args.exp_id % index.shape[1]]
        ref_ids = Info['ref_ids'][0, :]  #

        K = args.K_fold
        k = args.k_test
        testindex = index[int((k-1)/K * len(index)):int(k/K * len(index))]
        valindex = index[range(-int((5-k)/K * len(index)), -int((4-k)/K * len(index)))]
        train_index, val_index, test_index = [], [], []
        for i in range(len(ref_ids)):
            if ref_ids[i] in testindex:
                test_index.append(i)
            elif ref_ids[i] in valindex:
                val_index.append(i)
                train_index.append(i)  #
            else:
                train_index.append(i)
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

        self.scale = Info['subjective_scores'][0, :].max()
        self.mos = Info['subjective_scores'][0, self.index] / self.scale #
        self.mos_std = Info['subjective_scoresSTD'][0, self.index] / self.scale #
        im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()[::2].decode() for i in self.index]
        ref_names = [Info[Info['ref_names'][0, :][i]][()].tobytes()[::2].decode()
                     for i in (ref_ids[self.index]-1).astype(int)]

        self.patches = ()
        self.label = []
        self.label_std = []
        self.ims = []
        self.refs = []
        for idx in range(len(self.index)):
            # print("Preprocessing Image: {}".format(im_names[idx]))
            im = loader(os.path.join(args.im_dir, im_names[idx]))
            if args.ref_dir is None or 'NR' in args.model:
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
                # patches = RandomCropPatches(im, ref, args.patch_size, args.n_patches)
                self.patches = self.patches + (patches,)  #

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if self.status == 'train':
            patches = RandomCropPatches(self.ims[idx], self.refs[idx], self.patch_size, self.n_patches)
        else:
            patches = self.patches[idx]
        return patches, (torch.Tensor([self.label[idx], ]), torch.Tensor([self.label_std[idx], ]))


def mkdirs(path):
    # if not os.path.exists(path):
    #     os.makedirs(path)
    os.makedirs(path, exist_ok=True)


class FRnet(nn.Module):
    """
    (Wa)DIQaM-FR Model
    """
    def __init__(self, weighted_average=True):
        """
        :param weighted_average: weighted average or not?
        """
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
        self.fc1_q  = nn.Linear(512*3, 512)
        self.fc2_q  = nn.Linear(512, 1)
        self.fc1_w  = nn.Linear(512*3, 512)
        self.fc2_w  = nn.Linear(512, 1)
        self.dropout = nn.Dropout()
        self.weighted_average = weighted_average

    def extract_features(self, x):
        """
        feature extraction
        :param x: the input image
        :return: the output feature
        """
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

        h = h.view(-1, 512)

        return h

    def forward(self, data):
        """
        :param data: distorted and reference patches of images
        :return: quality of images/patches
        """
        x, x_ref = data
        batch_size = x.size(0)
        n_patches = x.size(1)
        if self.weighted_average:
            q = torch.ones((batch_size, 1), device=x.device)
        else:
            q = torch.ones((batch_size * n_patches, 1), device=x.device)

        for i in range(batch_size):

            h = self.extract_features(x[i])
            h_ref = self.extract_features(x_ref[i])
            h = torch.cat((h - h_ref, h, h_ref), 1)

            h_ = h  # save intermediate features

            h = F.relu(self.fc1_q(h_))
            h = self.dropout(h)
            h = self.fc2_q(h)

            if self.weighted_average:
                w = F.relu(self.fc1_w(h_))
                w = self.dropout(w)
                w = F.relu(self.fc2_w(w)) + 0.000001 # small constant
                q[i] = torch.sum(h * w) / torch.sum(w)
            else:
                q[i*n_patches:(i+1)*n_patches] = h

        return q


class NRnet(nn.Module):
    """
    (Wa)DIQaM-NR Model
    """
    def __init__(self, weighted_average=True):
        """
        :param weighted_average: weighted average or not?
        """
        super(NRnet, self).__init__()
        self.conv1   = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2   = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3   = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4   = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5   = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6   = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7   = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8   = nn.Conv2d(256, 256, 3, padding=1)
        self.conv9   = nn.Conv2d(256, 512, 3, padding=1)
        self.conv10  = nn.Conv2d(512, 512, 3, padding=1)
        self.fc1q_nr = nn.Linear(512, 512)
        self.fc2q_nr = nn.Linear(512, 1)
        self.fc1w_nr = nn.Linear(512, 512)
        self.fc2w_nr = nn.Linear(512, 1)
        self.dropout = nn.Dropout()
        self.weighted_average = weighted_average

    def extract_features(self, x):
        """
        feature extraction
        :param x: the input image
        :return: the output feature
        """
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

    def forward(self, x):
        """
        :param data: distorted and reference patches of images
        :return: quality of images/patches
        """
        batch_size = x.size(0)
        n_patches = x.size(1)
        if self.weighted_average:
            q = torch.ones((batch_size, 1), device=x.device)
        else:
            q = torch.ones((batch_size * n_patches, 1), device=x.device)

        for i in range(batch_size):

            h = self.extract_features(x[i])

            h_ = h  # save intermediate features

            h = F.relu(self.fc1q_nr(h_))
            h = self.dropout(h)
            h = self.fc2q_nr(h)

            if self.weighted_average:
                w = F.relu(self.fc1w_nr(h_))
                w = self.dropout(w)
                w = F.relu(self.fc2w_nr(w)) + 0.000001  # small constant
                q[i] = torch.sum(h * w) / torch.sum(w)
            else:
                q[i * n_patches:(i + 1) * n_patches] = h

        return q


class IQALoss(torch.nn.Module):
    def __init__(self):
        super(IQALoss, self).__init__()

    def forward(self, y_pred, y):
        """
        loss function, e.g., l1 loss
        :param y_pred: predicted values
        :param y: y[0] is the ground truth label
        :return: the calculated loss
        """
        n = int(y_pred.size(0) / y[0].size(0))  # n=1 if images; n>1 if patches
        loss = F.l1_loss(y_pred, y[0].repeat((1, n)).reshape((-1, 1)))  #
        return loss


class IQAPerformance(Metric):
    """
    Evaluation of IQA methods using SROCC, KROCC, PLCC, RMSE, MAE.

    `update` must receive output of the form (y_pred, y).
    """
    def reset(self):
        self._y_pred = []
        self._y      = []
        self._y_std  = []

    def update(self, output):
        y_pred, y = output

        self._y.append(y[0].item())
        self._y_std.append(y[1].item())
        n = int(y_pred.size(0) / y[0].size(0))  # n=1 if images; n>1 if patches
        y_pred_im = y_pred.reshape((y[0].size(0), n)).mean(dim=1, keepdim=True)
        self._y_pred.append(y_pred_im.item())

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))
        sq_std = np.reshape(np.asarray(self._y_std), (-1,))
        q = np.reshape(np.asarray(self._y_pred), (-1,))

        srocc = stats.spearmanr(sq, q)[0]
        krocc = stats.stats.kendalltau(sq, q)[0]
        plcc = stats.pearsonr(sq, q)[0]
        rmse = np.sqrt(((sq - q) ** 2).mean())
        mae = np.abs((sq - q)).mean()
        outlier_ratio = (np.abs(sq - q) > 2 * sq_std).mean()

        return srocc, krocc, plcc, rmse, mae, outlier_ratio


def get_data_loaders(args):
    """ Prepare the train-val-test data
    :param args: related arguments
    :return: train_loader, val_loader, test_loader, scale
    """
    train_dataset = IQADataset_less_memory(args, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4)  #

    val_dataset = IQADataset(args, 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset)

    test_dataset = IQADataset(args, 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset)

    scale = test_dataset.scale

    return train_loader, val_loader, test_loader, scale


def run(args):
    """
    Run the program
    """
    train_loader, val_loader, test_loader, scale = get_data_loaders(args)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    lr_ratio = 1  # feature lr / regression lr
    if args.model == 'WaDIQaM-FR' or args.model == 'DIQaM-FR':
        model = FRnet(weighted_average=args.weighted_average)
        if args.resume is not None:
            model.load_state_dict(torch.load(args.resume))
    elif args.model == 'WaDIQaM-NR' or args.model == 'DIQaM-NR':
        model = NRnet(weighted_average=args.weighted_average)
        if args.resume is not None:
            model_dict = model.state_dict()
            if 'FR' in args.resume:
                lr_ratio = 0.1  # set feature lr / regression lr to 1/10
                # Initialize the feature extractor by pretrained FRNet
                pretrained_model = FRnet(weighted_average=args.weighted_average)
                pretrained_model.load_state_dict(torch.load(args.resume))
                pretrained_dict = pretrained_model.state_dict()
                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)
    else:
        print('Wrong model name!')

    writer = SummaryWriter(log_dir=args.log_dir)
    model = model.to(device)
    print(model)

    if args.multi_gpu and torch.cuda.device_count() > 1:
        print("Using multiple GPU")
        model = nn.DataParallel(model)
        # batch_size becomes batch_size * torch.cuda.device_count()

        all_params = model.module.parameters()
        regression_params = []
        for pname, p in model.module.named_parameters():
            if pname.find('fc') >= 0:
                regression_params.append(p)
        regression_params_id = list(map(id, regression_params))
        features_params = list(filter(lambda p: id(p) not in regression_params_id, all_params))
        optimizer = Adam([{'params': regression_params},
                          {'params':  features_params, 'lr': args.lr*lr_ratio}],
                         lr=args.lr, weight_decay=args.weight_decay)
    else:
        all_params = model.parameters()
        regression_params = []
        for pname, p in model.named_parameters():
            if pname.find('fc') >= 0:
                regression_params.append(p)
        regression_params_id = list(map(id, regression_params))
        features_params = list(filter(lambda p: id(p) not in regression_params_id, all_params))
        optimizer = Adam([{'params': regression_params},
                          {'params':  features_params, 'lr': args.lr*lr_ratio}],
                         lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
    global best_criterion
    best_criterion = -1  # SROCC >= -1
    trainer = create_supervised_trainer(model, optimizer, IQALoss(), device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'IQA_performance': IQAPerformance()},
                                            device=device)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        writer.add_scalar("training/loss", scale * engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
        print("Validation Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
              .format(engine.state.epoch, SROCC, KROCC, PLCC, scale * RMSE, scale * MAE, 100 * OR))
        writer.add_scalar("SROCC/validation", SROCC, engine.state.epoch)
        writer.add_scalar("KROCC/validation", KROCC, engine.state.epoch)
        writer.add_scalar("PLCC/validation", PLCC, engine.state.epoch)
        writer.add_scalar("RMSE/validation", scale * RMSE, engine.state.epoch)
        writer.add_scalar("MAE/validation", scale * MAE, engine.state.epoch)
        writer.add_scalar("OR/validation", OR, engine.state.epoch)

        scheduler.step(engine.state.epoch)

        global best_criterion
        global best_epoch
        if SROCC > best_criterion and engine.state.epoch/args.epochs > 1/6:  #
        # if engine.state.epoch/args.epochs > 1/6 and engine.state.epoch % int(args.epochs/10) == 0:
            best_criterion = SROCC
            best_epoch = engine.state.epoch
            try:
                torch.save(model.module.state_dict(), args.trained_model_file)
            except:
                torch.save(model.state_dict(), args.trained_model_file)
                # torch.save(model.state_dict(), args.trained_model_file + str(engine.state.epoch))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_testing_results(engine):
        if args.test_during_training:
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
            print("Testing Results    - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
                  .format(engine.state.epoch, SROCC, KROCC, PLCC, scale * RMSE, scale * MAE, 100 * OR))
            writer.add_scalar("SROCC/testing", SROCC, engine.state.epoch)
            writer.add_scalar("KROCC/testing", KROCC, engine.state.epoch)
            writer.add_scalar("PLCC/testing", PLCC, engine.state.epoch)
            writer.add_scalar("RMSE/testing", scale * RMSE, engine.state.epoch)
            writer.add_scalar("MAE/testing", scale * MAE, engine.state.epoch)
            writer.add_scalar("OR/testing", OR, engine.state.epoch)

    @trainer.on(Events.COMPLETED)
    def final_testing_results(engine):
        global best_epoch
        model.load_state_dict(torch.load(args.trained_model_file))
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
        print("Final Test Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
            .format(best_epoch, SROCC, KROCC, PLCC, scale * RMSE, scale * MAE, 100 * OR))
        np.save(args.save_result_file, (SROCC, KROCC, PLCC, scale * RMSE, scale * MAE, OR))

    # kick everything off
    trainer.run(train_loader, max_epochs=args.epochs)

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch (Wa)DIQaM-FR/NR')
    parser.add_argument("--seed", type=int, default=19920517)
    # training parameters
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='number of epochs to train (default: 3000)')
    parser.add_argument('--decay_interval', type=int, default=100,
                        help='learning rate decay interval (default: 100)')
    parser.add_argument('--decay_ratio', type=int, default=0.8,
                        help='learning rate decay ratio (default: 0.8)')

    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('--K_fold', type=int, default=5,
                        help='K-fold cross-validation (default: 5)')
    parser.add_argument('--k_test', type=int, default=5,
                        help='The k-th fold used for test (1:K-fold, default: 5)')  # last 20%

    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--disable_visualization", action='store_true',
                        help='flag whether to disable TensorBoard visualization')
    parser.add_argument("--test_during_training", action='store_true',
                        help='flag whether to test during training')
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='flag whether to use multiple GPUs')
    # data info
    parser.add_argument('--database', default='LIVE', type=str,
                        help='database name (default: LIVE)')
    # model info
    parser.add_argument('--model', default='WaDIQaM-FR', type=str,
                        help='model name ((Wa)DIQaM-FR/NR, default: WaDIQaM-FR)')

    args = parser.parse_args()

    args.patch_size = 32  #
    args.n_patches = 32  #
    if 'Wa' in args.model:
        args.weighted_average = True
    else:
        args.weighted_average = False

    if args.database == 'TID2013':
        args.data_info = './data/TID2013fullinfo.mat'
        args.im_dir = '/media/ldq/Research/Data/tid2013/distorted_images/'
        args.ref_dir = '/media/ldq/Research/Data/tid2013/reference_images/'

    if args.database == 'TID2008':
        args.data_info = './data/TID2008fullinfo.mat'
        args.im_dir = '/media/ldq/Research/Data/tid2008/distorted_images/'
        args.ref_dir = '/media/ldq/Research/Data/tid2008/reference_images/'

    if args.database == 'KADID-10K':
        args.data_info = './data/KADID-10K.mat'
        args.im_dir = '/media/ldq/Others/Data/kadid10k/image/'
        args.ref_dir = '/media/ldq/Others/Data/kadid10k/image/'

    if args.database == 'LIVE':
        args.data_info = './data/LIVEfullinfo.mat'
        args.im_dir = '/media/ldq/Research/Data/databaserelease2/'
        args.ref_dir = '/media/ldq/Research/Data/databaserelease2/refimgs'

    if args.database == 'CLIVE':
        args.data_info = './data/CLIVEinfo.mat'
        args.im_dir = '/media/ldq/Research/Data/ChallengeDB_release/Images/'
        args.ref_dir = None

    if args.database == 'BID':
        args.data_info = './data/BIDinfo.mat'
        args.im_dir = '/media/ldq/Research/Data/BID/ImageDatabase/'
        args.ref_dir = None

    args.log_dir = '{}/EXP{}-{}-{}-{}-lr={}-bs={}'.format(args.log_dir, args.exp_id, args.k_test, args.database,
                                                          args.model, args.lr, args.batch_size)

    mkdirs('checkpoints')
    args.trained_model_file = 'checkpoints/{}-{}-EXP{}-{}-lr={}-bs={}'.format(args.model, args.database, args.exp_id,
                                                                              args.k_test, args.lr, args.batch_size)
    mkdirs('results')
    args.save_result_file = 'results/{}-{}-EXP{}-{}-lr={}-bs={}'.format(args.model, args.database, args.exp_id,
                                                                        args.k_test, args.lr, args.batch_size)

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    run(args)
