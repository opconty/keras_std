#-*- coding:utf-8 -*-
#'''
# Created on 19-5-13 下午3:46
#
# @Author: Greg Gao(laygin)
#'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from utils.bbox_process import DetectByCenterMap
from data import IcdarConfig, resize_image
from data.data_utils import preprocess_img



cfg = IcdarConfig()
score = 0.1
center_line_score = 0.5
max_size = 768  # 608,768
resize_version = 0
area_cont = 10
center_line_stride = 1

resize = True
vggmode = 'deconv'  # only for vgg16,  'deconv', 'upsample'
mode = 'caffe'

# config visualization
plot_centerline = True
vis_center_mask = True

weights_path = 'path_to_your_pretrained_model'
assert os.path.exists(weights_path), Exception('weights path does not exist...')
model = load_model(weights_path)

Detector = DetectByCenterMap(cls_score=score,
                             center_line_stride=center_line_stride,
                             center_line_score=center_line_score,
                             area_cont=area_cont)


def get_boxes(model, img_path):
    img = cv2.imread(img_path)
    assert img is not None, 'image path does not exists'
    print(f'ori size: {img.shape}')
    w, h = resize_image(img, max_size)
    if resize:
        img = cv2.resize(img, dsize=(w, h))
        print(f'resize size: {img.shape}')
    img_dis = img.copy()
    img = preprocess_img(img[..., ::-1], mode=mode)
    preds = model.predict(np.expand_dims(img, 0))

    return preds, img_dis


def plot(img):
    plt.imshow(img)
    plt.show()


############ inference  ######################
for img_name in os.listdir(cfg.img_dir_test):
    img_path = os.path.join(cfg.img_dir_test, img_name)
    preds, img_dis = get_boxes(model, img_path)

    all_quads = Detector.detect(preds, img_dis.shape[:2])
    print('all_quads shape: ', np.array(all_quads).shape)

    # plot center line mask
    tcl_mask = Detector.make_center_mask()
    mask, _ = Detector.build_mask_from_mini_boxes(img_dis.shape[:2], Detector.get_miniboxes(img_dis.shape[:2]))
    mask = mask.astype(np.uint8)
    tcl_mask = ((mask > 0) * (tcl_mask > 0)).astype(np.uint8) * 255
    if plot_centerline:
        print('tcl mask shape:', tcl_mask.shape)
        plot(tcl_mask)

    # visualize center mask individual quads
    if vis_center_mask and len(all_quads):
        img_dis = Detector.draw_quads(img_dis, all_quads, color=(255, 0, 0))

    # blending
    alpha = 0.6
    channel = 2
    img_dis[..., channel] = cv2.addWeighted(img_dis[..., channel], alpha, tcl_mask, 1-alpha, 0)
    plot(img_dis[..., ::-1])



