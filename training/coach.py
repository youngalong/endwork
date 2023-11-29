#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 11:18
# @Author  : youngalone
# @File    : coach.py
# @Software: PyCharm
import os
from collections import defaultdict

import torch
import numpy as np

from lpips import LPIPS
from torchvision import transforms
from tqdm import tqdm, trange

from config import config
from config.config import checkpoints_dir
from criteria import l2_loss
from criteria.localitly_regularizer import SpaceRegularizer
from models.utils import init_e4e, load_old_g
from util import logger


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
        self.lpips_loss = LPIPS(net='alex').to(config.device).eval()

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

    def train(self):
        self.G.synthesis.train()
        self.G.mapping.train()

        use_ball_holder = True
        w_pivots = []
        images = []

        logger.info('计算初始反演')
        for fname, image in tqdm(self.dataset):
            image_name = fname
            w_pivot = self.get_inversion(image)
            w_pivots.append(w_pivot)
            images.append((image_name, image))

        self.G = self.G.to(config.device)

        logger.info('微调生成器')

        for _ in trange(config.max_pti_steps):
            step_loss_dict = defaultdict(list)

            for data, w_pivot in zip(images, w_pivots):
                image_name, image = data
                image = image.unsqueeze(0)

                real_images_batch = image.to(config.device)

                generated_images = self.forward(w_pivot)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                                               self.G, use_ball_holder, w_pivot)

                step_loss_dict['loss'].append(loss.item())
                step_loss_dict['l2_loss'].append(l2_loss_val.item())
                step_loss_dict['loss_lpips'].append(loss_lpips.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                use_ball_holder = config.training_step % config.locality_regularization_interval == 0

                config.training_step += 1

            log_dict = {}
            for key, losses in step_loss_dict.items():
                loss_mean = sum(losses) / len(losses)
                loss_max = max(losses)
                log_dict[f'losses_agg/{key}_mean'] = loss_mean
                log_dict[f'losses_agg/{key}_max'] = loss_max

        logger.info('训练完成')
        w_pivots = torch.cat(w_pivots)
        return w_pivots

    def get_inversion(self, image):
        if self.encoder == 'e4e':
            w_pivot = self.get_e4e_inversion(image)
        else:
            pass

        w_pivot = w_pivot.to(config.device)
        return w_pivot

    def get_e4e_inversion(self, image):
        new_image = self.e4e_image_transform(image).to(config.device)
        _, w = self.encoder_net(new_image.unsqueeze(0), randomize_noise=False, return_latents=True, resize=False,
                                input_code=False)
        return w

    def calc_loss(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch):
        loss = 0.0

        if config.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(generated_images, real_images)
            loss += l2_loss_val * config.pt_l2_lambda
        if config.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss_lpips = torch.squeeze(loss_lpips)
            loss += loss_lpips * config.pt_lpips_lambda

        if use_ball_holder and config.use_locality_regularization:
            ball_holder_loss_val = self.space_regularizer.space_regularizer_loss(new_G, w_batch, log_name)
            loss += ball_holder_loss_val

        return loss, l2_loss_val, loss_lpips

    def forward(self, w):
        generated_images = self.G.synthesis(w, noise_mode='const', force_fp32=True)

        return generated_images
