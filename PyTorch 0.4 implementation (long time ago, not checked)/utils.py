# Implemented by Dingquan Li
# Email: dingquanli@pku.edu.cn
# Date: 2019/3/5
import os


def mkdirs(path):
    # if not os.path.exists(path):
    #     os.makedirs(path)
    os.makedirs(path, exist_ok=True)