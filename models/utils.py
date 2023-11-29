#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 21:56
# @Author  : youngalone
# @File    : utils.py
# @Software: PyCharm
import pickle
from argparse import Namespace

import torch

from config import config
from config.config import e4e_predictor_path, stylegan2_ada_ffhq_predictor_path
from models.e4e.psp import pSp


def init_e4e():
    ckpt = torch.load(e4e_predictor_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = e4e_predictor_path
    opts = Namespace(**opts)
    e4e_inversion_net = pSp(opts)
    e4e_inversion_net = e4e_inversion_net.eval().to(config.device).requires_grad_(False)
    return e4e_inversion_net


def load_old_g():
    return load_g(stylegan2_ada_ffhq_predictor_path)


def load_g(file_path):
    with open(file_path, 'rb') as f:
        old_g = pickle.load(f)['G_ema'].to(config.device).eval()
        old_g = old_g.float()
    return old_g
