# (Wa)DIQaM-FR/NR
PyTorch 1.1 (with Python 3.6) implementation of the following paper:

Bosse S, Maniry D, Müller K R, et al. [Deep neural networks for no-reference and full-reference image quality assessment](https://ieeexplore.ieee.org/document/8063957). IEEE Transactions on Image Processing, 2018, 27(1): 206-219.

You can refer to the [chainer](https://chainer.org/) codes (only the test part) from the original authors: [dmaniry/deepIQA](https://github.com/dmaniry/deepIQA)

## Note
- The hyper-parameter or some other experimental settings are not the same as the paper described, e.g., nonoverlapping patches are considered for validation/test images instead of random selection. Readers can refer to the [paper](https://ieeexplore.ieee.org/document/8063957) for the exact settings of the original paper.
- Test demo or test cross dataset? Please refer to [test_cross_dataset.py](https://github.com/lidq92/WaDIQaM/blob/master/PyTorch%200.3%20implementation/test_cross_dataset.py). I will upload the new trained model and update this test file when I am free.
- Warning!. The performance on each database is not guaranteed using the default settings of the code. Reproduced results are welcomed to reported.
- If you do not have enough memory, then change slightly the code in IQADataset class. Specifically, read image in `__getitem__` instead of  `__init__`.