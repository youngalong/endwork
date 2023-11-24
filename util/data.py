#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/15 23:21
# @Author  : youngalone
# @File    : data.py
# @Software: PyCharm
import os

from util.logger import logger

IMG_EXTENSIONS = [".jpg", ".JPG",
                  ".png", ".PNG",
                  ".jpeg", ".JPEG"]


def make_dataset(dir_path):
    images = []
    logger.info('处理数据集...')
    assert os.path.isdir(dir_path), "%s 不是一个合法的文件夹路径" % dir_path
    for fname in sorted(os.listdir(dir_path)):
        # print(fname, end='\n')
        if is_image(fname):
            path = os.path.join(dir_path, fname)
            fname = fname.split(".")[0]
            images.append((fname, path))
    return images


def is_image(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
