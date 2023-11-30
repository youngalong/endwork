#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/15 23:11
# @Author  : youngalone
# @File    : train.py
# @Software: PyCharm
import json
import os
import torch

import click
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config import config
from config.config import image_size
from dataset.my_dataset import MyDataset
from models.utils import save_tuned_g
from training.coach import Coach
from util.alignment import crop_face, calc_alignment_coefficients
from util.data import make_dataset
from util.logger import logger
from util.logger import logger_init


@click.command()
@click.option("--input_folder", help="输入图片文件夹路径", type=str, required=True)
@click.option("--output_folder", help="输出图片文件夹路径", type=str, required=True)
@click.option("--username", help="微调对象名", type=str, required=True)
@click.option('--scale', default=1.0, type=float)
@click.option('--center_sigma', type=float, default=1.0)
@click.option('--xy_sigma', type=float, default=3.0)
@click.option('--encoder', help='潜码编码器', type=str, required=True)
def main(input_folder, output_folder, username, scale, center_sigma, xy_sigma, encoder):
    logger_init(input_folder, output_folder, username, scale, encoder)

    config.name = username
    files = make_dataset(input_folder)
    logger.info('Number of images: %d', len(files))

    logger.info('对齐图像...')
    crops, orig_images, quads = crop_face(files, scale, center_sigma, xy_sigma)
    logger.info('图像对齐完成')

    # crops[0].save('crops.jpg')
    # orig_images[0].save('orig_images.jpg')

    ds = MyDataset(crops, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))
    coach = Coach(ds, encoder)

    ws = coach.train()

    # logger.info(ws.size())  # [5 * 18 * 512]
    save_tuned_g(coach.G, ws, quads, config.name)

    inverse_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads]

    gen = coach.G.requires_grad_(False).eval()

    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'opts.json'), 'w') as f:
        json.dump(config, f)

    for i, (coeffs, crop, orig_image, w) in tqdm(
            enumerate(zip(inverse_transforms, crops, orig_images, ws)), total=len(ws)):
        w = w[None]
        with torch.no_grad():
            inversion = gen.synthesis(w, noise_mode='const', force_fp32=True)
            pivot = coach.original_G.synthesis(w, noise_mode='const', force_fp32=True)
            inversion = to_pil_image(inversion)
            pivot = to_pil_image(pivot)

        save_image(pivot, output_folder, 'pivot', i)
        save_image(inversion, output_folder, 'inversion', i)
        save_image(paste_image(coeffs, pivot, orig_image), output_folder, 'pivot_projected', i)
        save_image(paste_image(coeffs, inversion, orig_image), output_folder, 'inversion_projected', i)


def save_image(image: Image.Image, output_folder, image_name, image_index, ext='jpg'):
    if ext == 'jpeg' or ext == 'jpg':
        image = image.convert('RGB')
    folder = os.path.join(output_folder, image_name)
    os.makedirs(folder, exist_ok=True)
    image.save(os.path.join(folder, f'{image_index}.{ext}'))


def paste_image(coeffs, img, orig_image):
    pasted_image = orig_image.copy().convert('RGBA')
    projected = img.convert('RGBA').transform(orig_image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    pasted_image.paste(projected, (0, 0), mask=projected)
    return pasted_image


def to_pil_image(tensor: torch.Tensor) -> Image.Image:
    x = (tensor[0].permute(1, 2, 0) + 1) * 255 / 2
    x = x.detach().cpu().numpy()
    x = np.rint(x).clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)

if __name__ == '__main__':
    main()
