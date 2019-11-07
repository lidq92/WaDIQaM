"""
Test Cross Dataset
    For help
    ```bash
    python test_cross_dataset.py --help
    ```
 Date: 2018/9/20
"""

from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from main import RandomCropPatches, NonOverlappingCropPatches, FRnet
import numpy as np
import h5py, os


if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch WaDIQaM-FR test on the whole cross dataset')
    parser.add_argument("--dist_dir", type=str, default=None,
                        help="distorted images dir.")
    parser.add_argument("--ref_dir", type=str, default=None,
                        help="reference images dir.")
    parser.add_argument("--names_info", type=str, default=None,
                        help=".mat file that includes image names in the dataset.")
    parser.add_argument("--model_file", type=str, default='checkpoints/WaDIQaM-FR-KADID-10K-EXP1000-5-lr=0.0001-bs=4',
                        help="model file (default: checkpoints/WaDIQaM-FR-KADID-10K-EXP1000-5-lr=0.0001-bs=4)")
    parser.add_argument("--save_path", type=str, default='scores',
                        help="save path (default: scores)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FRnet(weighted_average=True).to(device)

    model.load_state_dict(torch.load(args.model_file))

    Info = h5py.File(args.names_info, 'r')
    im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()\
                        [::2].decode() for i in range(len(Info['im_names'][0, :]))]
    ref_names = [Info[Info['ref_names'][0, :][i]][()].tobytes()\
                        [::2].decode() for i in (Info['ref_ids'][0, :]-1).astype(int)]

    model.eval()
    scores = []   
    with torch.no_grad():
        for i in range(len(im_names)):
            im = Image.open(os.path.join(args.dist_dir, im_names[i])).convert('RGB')
            ref = Image.open(os.path.join(args.ref_dir, ref_names[i])).convert('RGB')
            # data = RandomCropPatches(im, ref)
            data = NonOverlappingCropPatches(im, ref)
            
            dist_patches = data[0].unsqueeze(0).to(device)
            ref_patches = data[1].unsqueeze(0).to(device)
            score = model((dist_patches, ref_patches))
            scores.append(score.item())
    np.save(args.save_path, scores)