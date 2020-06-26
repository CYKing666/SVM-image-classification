import random
import os
import numpy as np
import sys
from IPython import display
sys.path.append("..")


def make_text(path, train):
    """
    构建train.txt和test.txt
    path: string 数据路径 我用的是data
    train：boolean 判断是训练还是测试
    """
    if train:
        path_list0 = os.listdir(path + "train/0/")
        path_list1 = os.listdir(path + "train/1/")
        fullname = path + "train.txt"
        # fullname_1 = path + "train1.txt"
        with open(fullname,'w') as f:
            for name in path_list0:
                f.write(path + "train/" + "0/" + name + " " + "0" + "\n")
        # with open(fullname_1,'w') as f:
            for name in path_list1:
                f.write(path + "train/" + "1/" + name + " " + "1" + "\n")
    else:
        path_list0 = os.listdir(path + "test/0/")
        path_list1 = os.listdir(path + "test/1/")
        fullname = path + "test.txt"
        with open(fullname,'w') as f:
            for name in path_list0:
                f.write(path + "test/" + "0/" + name + " " + "0" + "\n")
            for name in path_list1:
                f.write(path + "test/" + "1/" + name + " " + "1" + "\n")
