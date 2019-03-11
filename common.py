# -*- coding: utf-8 -*-
# File: common.py
import random

import numpy as np
import cv2

from tensorpack.dataflow import RNGDataFlow
from tensorpack.dataflow.imgaug import transform, ImageAugmentor


class DataFromListOfDict(RNGDataFlow):
    def __init__(self, lst, keys, shuffle=False):
        self._lst = lst
        self._keys = keys
        self._shuffle = shuffle
        self._size = len(lst)

    def __len__(self):
        return self._size

    def __iter__(self):
        if self._shuffle:
            self.rng.shuffle(self._lst)
        for dic in self._lst:
            dp = [dic[k] for k in self._keys]
            yield dp


class RotateImg(ImageAugmentor):
    """
    Random rotate the image
    Mostly copied from - 
    https://github.com/Paperspace/DataAugmentationForObjectDetection
    """

    def __init__(self, max_angle=90, prob=0.5):
        """
        Args:
            max_angle (float): the image is rotated by a factor drawn randomly
                                from a range (-`max_angle`, `max_angle`).
            prob (float): probability of flip.
        """
        super(RotateImg, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        h, w = img.shape[:2]
        do = self._rand_range() < self.prob
        cx, cy = w // 2, h // 2
        angle = random.uniform(-self.max_angle, self.max_angle)

        return (do, cy, cx, angle, h, w)

    def _augment(self, img, param):
        do, _, _, angle, _, _ = param
        if do:
            ret = self._rotate_im(img, angle)
        else:
            ret = img
        return ret

    def _augment_coords(self, coords, param):
        #TODO should we tell the difference between masks and boxes?
        #TODO perhaps no if we also let the mask's polygon the same as the box!
        do, cy, cx, angle, h, w = param
        if do:
            bboxes = point8_to_box(coords)
            corners = self._get_corners(bboxes)
            corners = np.hstack((corners, bboxes[:, 4:]))
            corners[:, :8] = self._rotate_box(
                corners[:, :8], angle, cx, cy, h, w)
            new_bbox = self._get_enclosing_box(corners)

            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            scale_factor_x = nW / w
            scale_factor_y = nH / h
            new_bbox[:, :4] /= [scale_factor_x,
                                scale_factor_y, scale_factor_x, scale_factor_y]
            bboxes = self._clip_box(new_bbox, [0, 0, w, h], 0.2)
            return box_to_point8(bboxes)
        else:
            return coords

    def _rotate_im(self, image, angle):
        """Rotate the image.

        Rotate the image such that the rotated image is enclosed inside the tightest
        rectangle. The area not occupied by the pixels of the original image is colored
        black. 

        Parameters
        ----------

        image : numpy.ndarray
            numpy image

        angle : float
            angle by which the image is to be rotated

        Returns
        -------

        numpy.ndarray
            Rotated Image

        """
        # grab the dimensions of the image and then determine the
        # centre
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        image = cv2.warpAffine(image, M, (nW, nH))
        # https://stackoverflow.com/questions/43939645/width-and-height-parameters-in-opencv
        image = cv2.resize(image, (w, h))
        return image

    def _get_corners(self, bboxes):
        """Get corners of bounding boxes

        Parameters
        ----------

        bboxes: numpy.ndarray
            Numpy array containing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`

        returns
        -------

        numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      

        """
        width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
        height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

        x1 = bboxes[:, 0].reshape(-1, 1)
        y1 = bboxes[:, 1].reshape(-1, 1)

        x2 = x1 + width
        y2 = y1

        x3 = x1
        y3 = y1 + height

        x4 = bboxes[:, 2].reshape(-1, 1)
        y4 = bboxes[:, 3].reshape(-1, 1)

        corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

        return corners

    def _rotate_box(self, corners, angle, cx, cy, h, w):
        """Rotate the bounding box.

        Parameters
        ----------

        corners : numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

        angle : float
            angle by which the image is to be rotated

        cx : int
            x coordinate of the center of image (about which the box will be rotated)

        cy : int
            y coordinate of the center of image (about which the box will be rotated)

        h : int 
            height of the image

        w : int 
            width of the image

        Returns
        -------

        numpy.ndarray
            Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
        """

        corners = corners.reshape(-1, 2)
        corners = np.hstack(
            (corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        calculated = np.dot(M, corners.T).T

        calculated = calculated.reshape(-1, 8)

        return calculated

    def _get_enclosing_box(self, corners):
        """Get an enclosing box for ratated corners of a bounding box

        Parameters
        ----------

        corners : numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  

        Returns 
        -------

        numpy.ndarray
            Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`

        """
        x_ = corners[:, [0, 2, 4, 6]]
        y_ = corners[:, [1, 3, 5, 7]]

        xmin = np.min(x_, 1).reshape(-1, 1)
        ymin = np.min(y_, 1).reshape(-1, 1)
        xmax = np.max(x_, 1).reshape(-1, 1)
        ymax = np.max(y_, 1).reshape(-1, 1)

        final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

        return final

    def _clip_box(self, bbox, clip_box, alpha):
        """Clip the bounding boxes to the borders of an image

        Parameters
        ----------

        bbox: numpy.ndarray
            Numpy array containing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`

        clip_box: numpy.ndarray
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
        def bbox_area(bbox):
            return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

        ar_ = (bbox_area(bbox)) # could this be zero?
        x_min = np.maximum(bbox[:, 0], clip_box[0]).reshape(-1, 1)
        y_min = np.maximum(bbox[:, 1], clip_box[1]).reshape(-1, 1)
        x_max = np.minimum(bbox[:, 2], clip_box[2]).reshape(-1, 1)
        y_max = np.minimum(bbox[:, 3], clip_box[3]).reshape(-1, 1)

        bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))

        delta_area = ((ar_ - bbox_area(bbox)) / ar_)

        mask = (delta_area < (1 - alpha)).astype(int)

        bbox = bbox[mask == 1, :]

        return bbox


class CustomResize(transform.TransformAugmentorBase):
    """
    Try resizing the shortest edge to a certain number
    while avoiding the longest edge to exceed max_size.
    """

    def __init__(self, short_edge_length, max_size, interp=cv2.INTER_LINEAR):
        """
        Args:
            short_edge_length ([int, int]): a [min, max] interval from which to sample the
                shortest edge length.
            max_size (int): maximum allowed longest edge length.
        """
        super(CustomResize, self).__init__()
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())

    def _get_augment_params(self, img):
        h, w = img.shape[:2]
        size = self.rng.randint(
            self.short_edge_length[0], self.short_edge_length[1] + 1)
        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return transform.ResizeTransform(h, w, newh, neww, self.interp)


def box_to_point8(boxes):
    """
    Args:
        boxes: nx4
        i.e.
        x1, y1, x2, y2

    Returns:
        (nx4)x2
        i.e. (when n = 1)
            x1, y1
            x2, y2
            x1, y2
            x2, y1
    """
    b = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]]
    b = b.reshape((-1, 2))
    return b


def point8_to_box(points):
    """
    Args:
        points: (nx4)x2
    Returns:
        nx4 boxes (x1y1x2y2)
    """
    p = points.reshape((-1, 4, 2))
    minxy = p.min(axis=1)   # nx2
    maxxy = p.max(axis=1)   # nx2
    return np.concatenate((minxy, maxxy), axis=1)


def segmentation_to_mask(polys, height, width):
    """
    Convert polygons to binary masks.

    Args:
        polys: a list of nx2 float array. Each array contains many (x, y) coordinates.

    Returns:
        a binary matrix of (height, width)
    """
    polys = [p.flatten().tolist() for p in polys]
    assert len(polys) > 0, "Polygons are empty!"

    import pycocotools.mask as cocomask
    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle)


def clip_boxes(boxes, shape):
    """
    Args:
        boxes: (...)x4, float
        shape: h, w
    """
    orig_shape = boxes.shape
    boxes = boxes.reshape([-1, 4])
    h, w = shape
    boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], w)
    boxes[:, 3] = np.minimum(boxes[:, 3], h)
    return boxes.reshape(orig_shape)


def filter_boxes_inside_shape(boxes, shape):
    """
    Args:
        boxes: (nx4), float
        shape: (h, w)

    Returns:
        indices: (k, )
        selection: (kx4)
    """
    assert boxes.ndim == 2, boxes.shape
    assert len(shape) == 2, shape
    h, w = shape
    indices = np.where(
        (boxes[:, 0] >= 0) &
        (boxes[:, 1] >= 0) &
        (boxes[:, 2] <= w) &
        (boxes[:, 3] <= h))[0]
    return indices, boxes[indices, :]


try:
    import pycocotools.mask as cocomask

    # Much faster than utils/np_box_ops
    def np_iou(A, B):
        def to_xywh(box):
            box = box.copy()
            box[:, 2] -= box[:, 0]
            box[:, 3] -= box[:, 1]
            return box

        ret = cocomask.iou(
            to_xywh(A), to_xywh(B),
            np.zeros((len(B),), dtype=np.bool))
        # can accelerate even more, if using float32
        return ret.astype('float32')

except ImportError:
    from utils.np_box_ops import iou as np_iou  # noqa
