#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/20 13:12
# @Author  : youngalone
# @File    : logger.py
# @Software: PyCharm
# logger.py
import logging

# 初始化全局的 logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train')

# 可选：添加文件处理器
file_handler = logging.FileHandler('train.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


def logger_init(input_folder, output_folder, name, scale, encoder, generator):
    logging.basicConfig(level=logging.DEBUG)
    logger2 = logging.getLogger('init_log')
    file_handler2 = logging.FileHandler('train.log')
    file_handler2.setLevel(logging.DEBUG)
    logger2.addHandler(file_handler2)

    logger2.info('')
    logger2.info('****************************************************************************')
    logger2.info('** input_folder: %s                                                         ', input_folder)
    logger2.info('** output_folder: %s                                                        ', output_folder)
    logger2.info('** name: %s                                                                 ', name)
    logger2.info('** scale: %s                                                                ', scale)
    logger2.info('** encoder: %s                                                              ', encoder)
    logger2.info('** generator: %s                                                            ', generator)
    logger2.info('****************************************************************************')
    logger2.info('')


