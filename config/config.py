#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/15 23:18
# @Author  : youngalone
# @File    : config.py
# @Software: PyCharm
import torch

name = ""
image_size = 1024
device = "cuda:0" if torch.cuda.is_available else "cpu"
checkpoints_dir = './checkpoints'
training_step = 1

# 预训练模型
shape_predictor_path = 'autodl-tmp/pretrained_models/shape_predictor_68_face_landmarks.dat'
e4e_predictor_path = 'autodl-tmp/pretrained_models/e4e_ffhq_encode.pt'
ir_se50_predictor_path = 'autodl-tmp/pretrained_models/model_ir_se50.pth'
stylegan2_ada_ffhq_predictor_path = 'autodl-tmp/pretrained_models/ffhq.pkl'

# 正则化
regularizer_alpha = 30
regularizer_l2_lambda = 0.1
regularizer_lpips_lambda = 0.1
latent_ball_num_of_samples = 1
locality_regularization_interval = 1
use_locality_regularization = True

# LOSS
pt_l2_lambda = 1
pt_lpips_lambda = 1

first_inv_steps = 50
max_pti_steps = 50

# 优化器
pti_adam_beta1 = 0.9
pti_learning_rate = 3e-5
