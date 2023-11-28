#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 11:18
# @Author  : youngalone
# @File    : coach.py
# @Software: PyCharm
import os

from lpips import LPIPS
from torchvision import transforms

from config.config import checkpoints_dir
from criteria.localitly_regularizer import SpaceRegularizer
from models.utils import init_e4e, load_old_g


class Coach:
    def __init__(self, dataset, encoder):
        self.dataset = dataset
        self.encoder = encoder

        # TODO
        if self.encoder == 'e4e':
            self.encoder_net = init_e4e()
        elif self.encoder == 'pSp':
            pass

        self.e4e_image_transform = transforms.Resize((256, 256))
        # 感知损失
        self.lpips_loss = LPIPS(net='alex').to('cuda:0').eval()

        self.restart_training()

        self.checkpoint_dir = checkpoints_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def restart_training(self):
        # Initialize networks
        self.G = load_old_g()
        self.G.requires_grad_(True)

        self.original_G = load_old_g()

        self.space_regularizer = SpaceRegularizer(self.original_G, self.lpips_loss)
        self.optimizer = self.configure_optimizers()
