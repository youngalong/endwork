#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/15 23:18
# @Author  : youngalone
# @File    : config.py
# @Software: PyCharm

name = ""
image_size = 1024
checkpoints_dir = './checkpoints'

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
