#-*- coding:utf-8 -*-
#'''
# Created on 19-5-10 下午4:21
#
# @Author: Greg Gao(laygin)
#'''
import os


class Config(object):
    Name = None
    basedir = 'path/to/dataset/root'  # which contains all dataset, e.g. synthText, Icdar13
    assert os.path.exists(basedir)
    ProjDir = 'path/to/project'  # for me is /home/gaolijun/workspace/keras_std
    assert os.path.exists(ProjDir), 'project directory does not exists'

    checkpoints_dir = os.path.join(ProjDir, 'checkpoints')
    outputs_dir = os.path.join(ProjDir, 'outputs')

    # image channel-wise mean to subtract, the order is BGR
    img_channel_mean = [103.939, 116.779, 123.68]
    batch_size = 8
    input_size = 608
    epochs = 500
    stride_size = 4
    scale = 'h'
    center_line_stride = 1  # added in 19-6-17
    region = 2
    alpha = 0.999

