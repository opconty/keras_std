#-*- coding:utf-8 -*-
#'''
# Created on 19-5-15 上午11:48
#
# @Author: Greg Gao(laygin)
#'''
import numpy as np
import xml.etree.ElementTree as ET
from skimage.draw import polygon as drawpoly
from keras.applications.imagenet_utils import preprocess_input


# with each text instances
def readxmlv2(path):
    gtboxes = []
    img_file = ''
    xml = ET.parse(path)
    for elem in xml.iter():
        if 'filename' in elem.tag:
            img_file = elem.text
        text = []
        if 'text' in elem.tag:
            for aa in list(elem):
                if 'object' in aa.tag:
                    for attr in list(aa):
                        if 'bndbox' in attr.tag:
                            xmin = float(attr.find('xmin').text)
                            ymin = float(attr.find('ymin').text)
                            xmax = float(attr.find('xmax').text)
                            ymax = float(attr.find('ymax').text)

                            text.append((xmin, ymin, xmax, ymax))
            gtboxes.append(np.array(text))

    return gtboxes, img_file


class TextInstance(object):
    '''
    a single whole text box class
    '''
    def __init__(self, points):
        '''

        :param points: numpy array, a bunch of minibox, which element is [x1,y1,x2,y2] if quads is False
                    otherwise, is (n, 4, 2), each quad contains four points
        '''
        self.points = points

    def get_center_points(self):
        p0 = self.get_top_points()
        p1 = self.get_bottom_points()

        return (p0 + p1) / 2

    def get_rect_orientation(self):
        '''this is only for rectangle now, compute height and width ,the decide the orientation
        horizontal or vertical'''
        w = self.points[-1][2] - self.points[0][0]
        h = self.points[-1][-1] - self.points[0][1]
        if h > w:
            return 0  # vertical
        else:
            return 1  # horizontal

    def get_top_points(self):
        if self.get_rect_orientation() == 1:
            endpoint = self.points[-1][[2, 1]]  # x2, y1
        else:
            endpoint = self.points[-1][[0, 3]]  # x1, y2
        return np.concatenate([self.points[0][[0, 1]], endpoint])  # top end points, (x01,y01,x11, y11)

    def get_bottom_points(self):
        if self.get_rect_orientation() == 1:
            startpoint = self.points[0][[0, 3]]  # x1, y2
        else:
            startpoint = self.points[0][[2, 1]]  # x2, y1
        return np.concatenate([startpoint, self.points[-1][[2, 3]]])

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


def fill_polygon(mask, polygon, value):
    """
    fill polygon in the mask with value
    :param mask: input mask
    :param polygon: polygon to draw
    :param value: fill value
    """
    rr, cc = drawpoly(polygon[:, 1], polygon[:, 0], shape=(mask.shape[0], mask.shape[1]))
    mask[rr, cc] = value


def make_center_line_mask(top_points,
                          bottom_points,
                          center_points,
                          tcl_mask,
                          expand=0.3):
    c1 = center_points[[0, 1]]
    c2 = center_points[[2, 3]]
    top1 = top_points[[0, 1]]
    top2 = top_points[[2, 3]]
    bottom1 = bottom_points[[0, 1]]
    bottom2 = bottom_points[[2, 3]]
    p1 = c1 + (top1 - c1) * expand
    p2 = c1 + (bottom1 - c1) * expand
    p3 = c2 + (bottom2 - c2) * expand
    p4 = c2 + (top2 - c2) * expand
    polygon = np.stack([p1, p2, p3, p4])

    fill_polygon(tcl_mask, polygon, 1)


def gaussian(kernel):
    sigma = ((kernel - 1) * 0.5 -1) * 0.3 + 0.8
    s = 2*(sigma**2)
    dx = np.exp(-np.square(np.arange(kernel) - int(kernel/2)) / s)
    return np.reshape(dx, (-1, 1))


# this is the updated version for detecting box and center line
def generate_targets_v2(cfg, annots):
    '''

    :param cfg: dataset config
    :param annots: list of annotations,each element shape; (n, 4), n is number of mini-boxes
    :param center_line_version:0 for make_center_line_mask, 1 for make_center_line_mask_gaussian
    :return:
    '''
    bboxes = np.copy(annots)
    scale_map = np.zeros((cfg.input_size//cfg.stride_size,cfg.input_size//cfg.stride_size, 2))
    offset_map = np.zeros((cfg.input_size//cfg.stride_size, cfg.input_size//cfg.stride_size,3))
    center_map = np.zeros((cfg.input_size//cfg.stride_size, cfg.input_size//cfg.stride_size, 3))
    center_map[:, :, 1] = 1
    tcl_mask = np.zeros((cfg.input_size // cfg.center_line_stride, cfg.input_size // cfg.center_line_stride),
                        dtype=np.uint8)

    if len(bboxes):
        for text_idx in range(len(bboxes)):
            if len(bboxes[text_idx]) == 0:
                continue

            text = TextInstance(bboxes[text_idx]//cfg.center_line_stride)
            make_center_line_mask(text.get_top_points(),
                                  text.get_bottom_points(),
                                  text.get_center_points(),
                                  tcl_mask)

            region = cfg.region
            boxes = bboxes[text_idx] / cfg.stride_size
            boxes[boxes <= 0] = 0
            for idx in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[idx]
                x1_i, y1_i, x2_i, y2_i = [int(i) for i in boxes[idx]]
                cx, cy = int((x1 + x2) / 2), int((y1+y2) / 2)
                if x2-x1 <= 0 or y2-y1 <= 0:
                    continue

                dx, dy = gaussian(x2_i - x1_i), gaussian(y2_i - y1_i)
                gau_map = np.multiply(dy, dx.T)

                center_map[y1_i:y2_i, x1_i:x2_i, 0] = np.maximum(center_map[y1_i:y2_i, x1_i:x2_i, 0], gau_map)
                center_map[y1_i:y2_i, x1_i:x2_i, 1] = 1
                center_map[cy, cx, 2] = 1

                scale_map[cy - region:cy + region + 1, cx - region:cx + region + 1, 0] = np.log(y2-y1)
                scale_map[cy - region:cy + region + 1, cx - region:cx + region + 1, 1] = 1

                offset_map[cy, cx, 0] = (y1 + y2) / 2 - cy - 0.5
                offset_map[cy, cx, 1] = (x1 + x2) / 2 - cx - 0.5
                offset_map[cy, cx, 2] = 1

    return tcl_mask, center_map, scale_map, offset_map


def preprocess_img(rgb_img, mode='caffe'):
    return preprocess_input(rgb_img, mode=mode)


