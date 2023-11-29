#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/26 11:42
# @Author  : youngalone
# @File    : my_dataset.py
# @Software: PyCharm
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        name = str(index)
        image = image.convert('RGB')
        image = self.transform(image)

        return name, image
