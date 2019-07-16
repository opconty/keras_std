#-*- coding:utf-8 -*-
#'''
# Created on 19-5-13 下午4:09
#
# @Author: Greg Gao(laygin)
#'''
import cv2
import numpy as np


class DetectByCenterMap():
    '''
    generate center mask from center_cls, with or without center line prediction
    '''
    def __init__(self,
                 cls_score=0.1,
                 center_line_stride=1,
                 center_line_score=0.5,
                 center_line_version=0,
                 area_cont=0):
        self.cls_score = cls_score  # to filter mini boxes
        self.aspect_ratio = 1
        self.center_line_stride = center_line_stride
        self.center_line_score = center_line_score
        self.area_cont = area_cont
        self.center_line_version = center_line_version
        self.center_line = None
        self.center_cls = None
        self.scale_regr = None
        self.offset = None

    @staticmethod
    def is_in_contour(cont, point):
        x, y = point
        return cv2.pointPolygonTest(cont, (x, y), False) > 0

    def _extract_prediction(self, Y):
        self.center_line, self.center_cls, self.scale_regr, self.offset = Y

    def get_miniboxes(self, imgsize):
        stride_size = 4
        seman = self.center_cls[0, :, :, 0]
        height = self.scale_regr[0, :, :, 0]
        offset_y = self.offset[0, :, :, 0]
        offset_x = self.offset[0, :, :, 1]
        y_c, x_c = np.where(seman >= self.cls_score)
        boxs = []
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * stride_size
            w = self.aspect_ratio * h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * stride_size - w / 2), max(0, (
                        y_c[i] + o_y + 0.5) * stride_size - h / 2)
            boxs.append([x1, y1, min(x1 + w, imgsize[1]), min(y1 + h, imgsize[0]), s])
        boxs = np.array(boxs, dtype=np.float32)
        return boxs

    def make_center_mask(self):
        center_line = self.center_line[0,:,:,0]
        center_line = (center_line >= self.center_line_score).astype(np.uint8)
        center_mask = cv2.resize(center_line, dsize=None, fx=self.center_line_stride, fy=self.center_line_stride)
        return center_mask

    @staticmethod
    def build_mask_from_mini_boxes(img_size, boxes):
        mask = np.zeros(img_size, dtype=np.uint8)
        scores = []
        for b in boxes:
            x1, y1, x2, y2 = [int(i) for i in b[:4]]
            mask[y1:y2, x1:x2] = 255
            scores.append(b[-1])

        return mask, np.mean(scores)

    @staticmethod
    def _make_rect_from_miniboxes_of_text(imgsize, boxes):
        '''

        :param imgsize: img size (h, w)
        :param boxes: mini boxes within one text, (n, 5), which 5 denotes (xmin, ymin, xmax, ymax, score)
        :return: text level boxes, (top-left, top-right, bottom-right, bottom-left)
        '''
        qs = np.array(boxes)
        xmin = np.max(np.min(qs[:, 0]), 0)
        ymin = np.max(np.min(qs[:, 1]), 0)
        xmax = np.min(np.max(qs[:, 2]), imgsize[1])
        ymax = np.min(np.max(qs[:, 3]), imgsize[0])
        score = np.mean(qs[:, -1])
        return np.array([xmin, ymin, xmax, ymax]), score

    def classify_boxes_to_a_contour(self, miniboxes, contour):
        text = []
        for b in miniboxes:
            cx, cy = b[[0, 2]].sum() / 2, b[[1, 3]].sum() / 2
            if self.is_in_contour(contour, (cx, cy)):
                text.append(b)

        return text

    @staticmethod
    def _find_contours(mask, chain=cv2.CHAIN_APPROX_SIMPLE):
        try:
            _, conts, _ = cv2.findContours(mask, cv2.RETR_TREE, chain)
        except:
            conts, _ = cv2.findContours(mask, cv2.RETR_TREE, chain)

        return conts

    def _get_words_boxes_from_mask(self, mask):
        '''

        :param mask: binary mask
        :return: words level bounding boxes
        '''
        conts = self._find_contours(mask)
        rects = []
        for c in conts:
            r = cv2.minAreaRect(c)  # (x,y),(w,h), a = rect
            b = np.int0(cv2.boxPoints(r))   # left down, left up, right up, right down
            ld, lu, ru, rd = b
            rects.append([lu, ru, rd, ld])  # clockwise
        return np.array(rects)

    def detect(self, Y, imgsize:(tuple, list)):
        assert len(imgsize) == 2, 'img size must be a tuple or list which includes height and width'
        self._extract_prediction(Y)
        mini_boxes = self.get_miniboxes(imgsize)
        center_mask = self.make_center_mask()
        mask, _ = self.build_mask_from_mini_boxes(imgsize, mini_boxes)
        # make center line mask from mask and center score mask
        tcl_mask = ((mask>0) * (center_mask>0)).astype(np.uint8) * 255

        # find contours from tcl mask, to split mini boxes into different text lines
        conts = self._find_contours(tcl_mask)

        all_quads = []
        for c in conts:
            if cv2.contourArea(c) > self.area_cont:
                text = self.classify_boxes_to_a_contour(mini_boxes, c)
                if len(text):
                    text_mask, score = self.build_mask_from_mini_boxes(imgsize, text)
                    rects = self._get_words_boxes_from_mask(text_mask)

                    all_quads.extend(rects)

        if len(all_quads):
            all_quads = np.stack(all_quads)

        return all_quads

    @staticmethod
    def draw_quads(img_to_be_plotted, all_quads, color: tuple = (0, 0, 255)):
        if len(all_quads):
            img_to_be_plotted = cv2.drawContours(img_to_be_plotted, all_quads, -1, color, 2)

        return img_to_be_plotted
