#-*- coding:utf-8 -*-
#'''
# Created on 19-5-11 下午2:25
#
# @Author: Greg Gao(laygin)
#'''
import numpy as np
import cv2


'''resize image'''
def bbox_area(bbox):
    '''

    :param bbox: 2d array of xmin,ymin, xmax, ymax
    :return:
    '''
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def clip_box(bbox, clip_box, alpha=0.25):
    """Clip the bounding boxes to the borders of an image

    Parameters
    ----------

    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    clip_box: list, tuple
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`

    alpha: float
        If the fraction of a bounding box left in the image after being clipped is
        less than `alpha` the bounding box is dropped.

    Returns
    -------

    numpy.ndarray
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    ar_ = bbox_area(bbox)
    x_min = np.maximum(bbox[:, 0], clip_box[0]).reshape(-1, 1)
    y_min = np.maximum(bbox[:, 1], clip_box[1]).reshape(-1, 1)
    x_max = np.minimum(bbox[:, 2], clip_box[2]).reshape(-1, 1)
    y_max = np.minimum(bbox[:, 3], clip_box[3]).reshape(-1, 1)

    bbox = np.hstack((x_min, y_min, x_max, y_max))

    delta_area = ((ar_ - bbox_area(bbox)) / (ar_ + 1e-6))
    h = bbox[:,3] - bbox[:,1]
    w = bbox[:,2] - bbox[:,0]

    mask = (delta_area < (1 - alpha)) & (w > 0) & (h > 0) & (ar_ > 0)

    bbox = bbox[mask, :]

    return bbox


class RandomCrop(object):
    """Crop randomly the image in a sample.

        Args:
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
        """
    def __init__(self, output_size, verbose=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        self.verbose = verbose

    def __call__(self, image, coors):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        # left, top = 52, 9
        if self.verbose:
            print(self.__class__.__name__, left, top)

        image = image[top: top + new_h,
                left: left + new_w]

        for i in range(len(coors)):
            coors[i] -= [left, top, left, top]
            coors[i] = clip_box(coors[i], [0, 0, new_w, new_h], 0.25)

        return image, coors


class Resize(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, coors, size=None):
        h, w = image.shape[:2]
        if size is not None:
            self.output_size = size

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_w, new_h))
        for i in range(len(coors)):
            coors[i][:, [0, 2]] = coors[i][:, [0, 2]] * new_w / w
            coors[i][:, [1, 3]] = coors[i][:, [1, 3]] * new_h / h

        return img, coors


'''augmentation'''
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert 255 >= delta >= 0, 'delta is invalid'
        self.delta = delta

    def __call__(self, img, coors=None):
        img = img.astype(np.float32)
        if np.random.randint(0,2):
            delta = np.random.uniform(-self.delta, self.delta)
            img += delta
        return np.clip(img, 0, 255), coors


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, img, coors=None):
        img = img.astype(np.float32)
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            img *= alpha
        return np.clip(img, 0, 255), coors


# resize image such that image size should be divides exactly by 32 for inference
# this version image max size may not the max_img_size
def resize_image(im, max_img_size):
    im_width = np.minimum(im.shape[1], max_img_size)
    if im_width == max_img_size < im.shape[1]:
        im_height = int((im_width / im.shape[1]) * im.shape[0])
    else:
        im_height = im.shape[0]
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    d_width = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_width, d_height



