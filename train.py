#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/15 23:11
# @Author  : youngalone
# @File    : train.py
# @Software: PyCharm

#


# def main():
#     config.name = name
#     files = make_dataset(input_folder)
#     print(files)

import click

from config import config
from util.data import make_dataset
from util.logger import logger_init


@click.command()
@click.option("--input_folder", help="输入图片文件夹路径", type=str, required=True)
@click.option("--output_folder", help="输出图片文件夹路径", type=str, required=True)
@click.option("--username", help="微调对象名", type=str, required=True)
def main(input_folder, output_folder, username):
    logger_init(input_folder, output_folder, username)

    config.name = username
    files = make_dataset(input_folder)
    print(files)


if __name__ == '__main__':
    main()
