#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/15 23:11
# @Author  : youngalone
# @File    : train.py
# @Software: PyCharm

import click
from torchvision import transforms

from config import config
from dataset.my_dataset import MyDataset
from training.coach import Coach
from util.alignment import crop_face
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

    # ds = ImageListDataset(crops, transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    ds = MyDataset(crops, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))
    coach = Coach(ds, encoder)


if __name__ == '__main__':
    main()
