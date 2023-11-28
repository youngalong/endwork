#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 11:18
# @Author  : youngalone
# @File    : coach.py
# @Software: PyCharm
from models.utils import init_e4e


class Coach:
    def __init__(self, dataset, encoder):
        self.dataset = dataset
        self.encoder = encoder

        # TODO
        if self.encoder == 'e4e':
            self.encoder_net = init_e4e()
        elif self.encoder == 'pSp':
            pass
