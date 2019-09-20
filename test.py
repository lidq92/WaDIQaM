"""
Test
    For help
    ```bash
    python test.py --help
    ```
 Date: 2019/9/20
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
    parser = ArgumentParser(description='PyTorch WaDIQaM-FR test')
    parser.add_argument("--dist_path", type=str, default='images/img98_colorblock_5.jpg',
                        help="distorted image path.")
    parser.add_argument("--ref_path", type=str, default='images/img98.jpg',
                        help="reference image path.")
    parser.add_argument("--model_file", type=str, default='checkpoints/WaDIQaM-FR-KADID-10K-EXP1000-5-lr=0.0001-bs=4',
                        help="model file (default: checkpoints/WaDIQaM-FR-KADID-10K-EXP1000-5-lr=0.0001-bs=4)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FRnet(weighted_average=True).to(device)

    model.load_state_dict(torch.load(args.model_file))

    model.eval()
    with torch.no_grad():
        im = Image.open(args.dist_path).convert('RGB')
        ref = Image.open(args.ref_path).convert('RGB')
        # data = RandomCropPatches(im, ref)
        data = NonOverlappingCropPatches(im, ref)
        
        dist_patches = data[0].unsqueeze(0).to(device)
        ref_patches = data[1].unsqueeze(0).to(device)
        score = model((dist_patches, ref_patches))
        print(score.item())