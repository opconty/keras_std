#-*- coding:utf-8 -*-
#'''
# Created on 19-5-17 下午3:39
#
# @Author: Greg Gao(laygin)
#'''
import os
from keras.utils import Sequence
import numpy as np
import cv2
from config import Config
from .img_aug import RandomCrop, Resize
from .data_utils import generate_targets_v2, readxmlv2
from .data_utils import preprocess_img


class SynthTextConfig(Config):
    Name = 'synthText'
    data_dir = os.path.join(Config.basedir, 'SynthTextDetectionEnglish')
    assert os.path.exists(data_dir), 'directory not exists: {}'.format(data_dir)

    img_dir_train = os.path.join(data_dir, 'images')
    annot_dir_train = os.path.join(data_dir, 'std_annotations')

    batch_size = 4
    input_size = 384
    stride_size = 4
    scale = 'h'
    region = 2
    alpha = 0.999
    num_scale = 1
    model = 'vgg16'
    mode = 'caffe'  # preprocess mode
    shrink_ratio = 1.5


class SynthTextDataset(Sequence):
    def __init__(self, cfg: SynthTextConfig, shuffle=False, vis=False, augs=None):
        self.cfg = cfg
        self.shuffle = shuffle
        self.annot_dir = self.cfg.annot_dir_train
        self.img_dir = self.cfg.img_dir_train
        self.vis = vis
        self.augs = augs

        self.annot_names = os.listdir(self.annot_dir)
        self.si = self._size()
        self.random_crop = RandomCrop(self.cfg.input_size)
        self.resize = Resize((cfg.input_size, cfg.input_size))
        self.on_epoch_end()

    def __len__(self):
        return len(self.annot_names) // self.cfg.batch_size

    def _size(self):
        return len(self.annot_names)

    def __getitem__(self, idx):
        lb = idx * self.cfg.batch_size
        rb = (idx + 1) * self.cfg.batch_size

        if rb > self.si:
            rb = self.si
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

        b_tcl_mask = np.zeros((b, self.cfg.input_size//self.cfg.center_line_stride,
                               self.cfg.input_size//self.cfg.center_line_stride, 1), dtype=np.uint8)

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
            ann_path = os.path.join(self.annot_dir, ann_name)
            assert os.path.exists(ann_path), f'ann path : {ann_path} does not exist'

            bboxes, img_name = readxmlv2(ann_path)
            img_path = os.path.join(self.img_dir, img_name)

            img = cv2.imread(img_path)
            assert img is not None, f'img path does not exist:{img_name}'
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
            if self.augs:
                for a in self.augs:
                    img, bboxes = a(img, bboxes)

            img = img.astype(np.float32)
            if not self.vis:
                img = preprocess_img(img[..., ::-1], mode=self.cfg.mode)

            tcl_mask, center_map, scale_map, offset_map = generate_targets_v2(self.cfg, bboxes)
            return img, tcl_mask, center_map, scale_map, offset_map
        except Exception as e:
            import traceback
            traceback.print_exc()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.annot_names)

