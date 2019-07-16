#-*- coding:utf-8 -*-
#'''
# Created on 19-5-15 下午12:44
#
# @Author: Greg Gao(laygin)
#'''
import os
from keras.utils import Sequence
import numpy as np
import cv2
from config import Config
from .img_aug import RandomCrop, Resize
from .data_utils import generate_targets_v2, readxmlv2, preprocess_img
from data import SynthTextConfig


class IcdarConfig(Config):
    Name = 'icdar13'
    data_dir = os.path.join(Config.basedir, 'icdar2013')

    assert os.path.exists(data_dir), 'directory does not exists'

    img_dir_train = os.path.join(data_dir, 'images')
    annot_dir_train = os.path.join(data_dir, 'std_annotations')
    # testing set
    img_dir_test = os.path.join(data_dir, 'testing', 'images')

    batch_size = 4
    input_size = 384
    stride_size = 4
    scale = 'h'

    use_synthetic = False
    synthsize = 100

    region = 2
    alpha = 0.999
    num_scale = 1
    model = 'vgg16'
    mode = 'caffe'  # preprocess mode
    shrink_ratio = 1.5


class IcdarDataset(Sequence):
    def __init__(self, cfg: IcdarConfig, shuffle=False, augments: (list, tuple)=None):
        self.cfg = cfg
        self.shuffle = shuffle
        self.datasets = []  # sample from synthetic data
        self.syn_cfg = SynthTextConfig()
        if self.cfg.use_synthetic:
            self.datasets = [(self.cfg.img_dir_train, self.cfg.annot_dir_train, os.listdir(self.cfg.annot_dir_train)),  # icdar
                             (self.syn_cfg.img_dir_train,
                              self.syn_cfg.annot_dir_train,
                              np.random.choice(os.listdir(self.syn_cfg.annot_dir_train), self.cfg.synthsize,
                                               replace=False).tolist())]
        else:
            self.annot_dir = self.cfg.annot_dir_train
            self.img_dir = self.cfg.img_dir_train

        self.augments = augments
        self.annot_names = []
        if self.cfg.use_synthetic:
            for img_dir, ann_dir, annos in self.datasets:
                for a in annos:
                    self.annot_names.append([img_dir, ann_dir, a])
            np.random.shuffle(self.annot_names)
        else:
            self.annot_names = os.listdir(self.annot_dir)

        self.random_crop = RandomCrop(self.cfg.input_size)
        self.resize = Resize((cfg.input_size, cfg.input_size))
        self.on_epoch_end()

    def __len__(self):
        return len(self.annot_names) // self.cfg.batch_size

    def __getitem__(self, idx):
        lb = idx * self.cfg.batch_size
        rb = (idx + 1) * self.cfg.batch_size
        si = len(self.annot_names)
        if rb > si:
            rb = si
            lb = rb - self.cfg.batch_size
        b = rb - lb

        b_img = np.zeros(
            (b, self.cfg.input_size, self.cfg.input_size, 3), dtype=np.float32)
        b_center_map = np.zeros(
            (b, self.cfg.input_size//self.cfg.stride_size, self.cfg.input_size//self.cfg.stride_size, 3))
        b_scale_map = np.zeros(
            (b, self.cfg.input_size // self.cfg.stride_size, self.cfg.input_size // self.cfg.stride_size, 2))
        b_offset_map = np.zeros(
            (b, self.cfg.input_size // self.cfg.stride_size, self.cfg.input_size // self.cfg.stride_size, 3))

        b_tcl_mask = np.zeros((b, self.cfg.input_size // self.cfg.center_line_stride,
                               self.cfg.input_size // self.cfg.center_line_stride, 1), dtype=np.uint8)

        for i, ann_name in enumerate(self.annot_names[lb:rb]):
            a = self._aug_img(ann_name)
            if a is not None:
                img, tcl_mask, center_map, scale_map, offset_map = a

                b_img[i] = img
                b_center_map[i] = center_map
                b_scale_map[i] = scale_map
                if offset_map is not None:
                    b_offset_map[i] = offset_map
                    b_tcl_mask[i] = np.expand_dims(tcl_mask, axis=-1)

            else:
                print(ann_name)
                continue

        return [b_img], [b_tcl_mask, b_center_map, b_scale_map, b_offset_map]

    def _aug_img(self, ann_name):
        try:
            if self.cfg.use_synthetic:
                self.img_dir, self.annot_dir, ann_na = ann_name
                ann_path = os.path.join(self.annot_dir, ann_na)
            else:
                ann_path = os.path.join(self.annot_dir, ann_name)
            assert os.path.exists(ann_path), f'ann path : {ann_path} does not exist'

            bboxes, img_name = readxmlv2(ann_path)

            img_path = os.path.join(self.img_dir, img_name)

            img = cv2.imread(img_path)
            assert img is not None, f'img path does not exists, {img_path}'
            h, w = img.shape[:2]
            # due to bboxes may be None, using try...except

            # if image minimum larger than target size, crop it directly
            # elif image maximum larger than target size, resize it to maximum then crop
            # else resize to n times target size and crop, which n larger than 1
            minimum = np.minimum(h, w)
            maximum = np.maximum(h, w)
            resize_size = None
            if self.cfg.input_size * self.cfg.shrink_ratio < minimum:
                resize_size = self.cfg.input_size * self.cfg.shrink_ratio
            elif maximum > self.cfg.input_size >= minimum:
                resize_size = maximum
            elif self.cfg.input_size >= maximum:
                resize_size = int(self.cfg.shrink_ratio * self.cfg.input_size)

            if resize_size:
                img, bboxes = self.resize(img, bboxes, size=int(resize_size))

            img, bboxes = self.random_crop(img, bboxes)

            # apply augments
            if self.augments:
                for a in self.augments:
                    img, bboxes = a(img, bboxes)

            img = img.astype(np.float32)
            img = preprocess_img(img[..., ::-1], mode=self.cfg.mode)

            tcl_mask, center_map, scale_map, offset_map = generate_targets_v2(self.cfg,
                                                                              bboxes)
            return img, tcl_mask, center_map, scale_map, offset_map

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.annot_names)

