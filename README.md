# (Wa)DIQaM-FR/NR
PyTorch 1.1 (with Python 3.6) implementation of the following paper:

Bosse S, Maniry D, Müller K R, et al. [Deep neural networks for no-reference and full-reference image quality assessment](https://ieeexplore.ieee.org/document/8063957). IEEE Transactions on Image Processing, 2018, 27(1): 206-219.

You can refer to the [chainer](https://chainer.org/) codes (only the test part) from the original authors: [dmaniry/deepIQA](https://github.com/dmaniry/deepIQA)

## Note
- The hyper-parameter or some other experimental settings are not the same as the paper described, e.g., nonoverlapping patches are considered for validation/test images instead of random selection. Readers can refer to the [paper](https://ieeexplore.ieee.org/document/8063957) for the exact settings of the original paper.
- Warning!. The performance on each database is not guaranteed using the default settings of the code. Reproduced results are welcomed to reported.
- If you do not have enough memory, then change slightly the code in `IQADataset` class. Specifically, read image in `__getitem__` instead of  `__init__`. You can choose to use `IQADataset_less_memory` class instead.

## TODO (If I have free time)
- Reproduce the results on some common databases, especially for the NR model (Currently, NR model is not tuned to reproduce the results.)
- Simplify the code
- etc.