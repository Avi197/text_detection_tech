# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple
from collections import abc
from typing import Any, Callable, Optional, Type, Union
import numpy as np

ArrayLike = 'ArrayLike'


def bbox2poly(bbox: ArrayLike, mode: str = 'xyxy') -> np.array:
    """Converting a bounding box to a polygon.

    Args:
        bbox (ArrayLike): A bbox. In any form can be accessed by 1-D indices.
         E.g. list[float], np.ndarray, or torch.Tensor. bbox is written in
            [x1, y1, x2, y2].
        mode (str): Specify the format of bbox. Can be 'xyxy' or 'xywh'.
            Defaults to 'xyxy'.

    Returns:
        np.array: The converted polygon [x1, y1, x2, y1, x2, y2, x1, y2].
    """
    assert len(bbox) == 4
    if mode == 'xyxy':
        x1, y1, x2, y2 = bbox
        poly = np.array([x1, y1, x2, y1, x2, y2, x1, y2])
    elif mode == 'xywh':
        x, y, w, h = bbox
        poly = np.array([x, y, x + w, y, x + w, y + h, x, y + h])
    else:
        raise NotImplementedError('Not supported mode.')

    return poly


def poly2bbox(polygon: ArrayLike) -> np.array:
    """Converting a polygon to a bounding box.

    Args:
         polygon (ArrayLike): A polygon. In any form can be converted
             to an 1-D numpy array. E.g. list[float], np.ndarray,
             or torch.Tensor. Polygon is written in
             [x1, y1, x2, y2, ...].

     Returns:
         np.array: The converted bounding box [x1, y1, x2, y2]
    """
    assert len(polygon) % 2 == 0
    polygon = np.array(polygon, dtype=np.float32)
    x = polygon[::2]
    y = polygon[1::2]
    return np.array([min(x), min(y), max(x), max(y)])


def is_on_same_line(box_a, box_b, min_y_overlap_ratio=0.8):
    # TODO Check if it should be deleted after ocr.py refactored
    """Check if two boxes are on the same line by their y-axis coordinates.

    Two boxes are on the same line if they overlap vertically, and the length
    of the overlapping line segment is greater than min_y_overlap_ratio * the
    height of either of the boxes.

    Args:
        box_a (list), box_b (list): Two bounding boxes to be checked
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                                    allowed for boxes in the same line

    Returns:
        The bool flag indicating if they are on the same line
    """
    a_y_min = np.min(box_a[1::2])
    b_y_min = np.min(box_b[1::2])
    a_y_max = np.max(box_a[1::2])
    b_y_max = np.max(box_b[1::2])

    # Make sure that box a is always the box above another
    if a_y_min > b_y_min:
        a_y_min, b_y_min = b_y_min, a_y_min
        a_y_max, b_y_max = b_y_max, a_y_max

    if b_y_min <= a_y_max:
        if min_y_overlap_ratio is not None:
            sorted_y = sorted([b_y_min, b_y_max, a_y_max])
            overlap = sorted_y[1] - sorted_y[0]
            min_a_overlap = (a_y_max - a_y_min) * min_y_overlap_ratio
            min_b_overlap = (b_y_max - b_y_min) * min_y_overlap_ratio
            return overlap >= min_a_overlap or \
                   overlap >= min_b_overlap
        else:
            return True
    return False


def stitch_boxes_into_lines(boxes, max_x_dist=10, min_y_overlap_ratio=0.8):
    # TODO Check if it should be deleted after ocr.py refactored
    """Stitch fragmented boxes of words into lines.

    Note: part of its logic is inspired by @Johndirr
    (https://github.com/faustomorales/keras-ocr/issues/22)

    Args:
        boxes (list): List of ocr results to be stitched
        max_x_dist (int): The maximum horizontal distance between the closest
                    edges of neighboring boxes in the same line
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                    allowed for any pairs of neighboring boxes in the same line

    Returns:
        merged_boxes(list[dict]): List of merged boxes and texts
    """

    if len(boxes) <= 1:
        return boxes

    merged_boxes = []

    # sort groups based on the x_min coordinate of boxes
    x_sorted_boxes = sorted(boxes, key=lambda x: np.min(x['box'][::2]))
    # store indexes of boxes which are already parts of other lines
    skip_idxs = set()

    i = 0
    # locate lines of boxes starting from the leftmost one
    for i in range(len(x_sorted_boxes)):
        if i in skip_idxs:
            continue
        # the rightmost box in the current line
        rightmost_box_idx = i
        line = [rightmost_box_idx]
        for j in range(i + 1, len(x_sorted_boxes)):
            if j in skip_idxs:
                continue
            if is_on_same_line(x_sorted_boxes[rightmost_box_idx]['box'],
                               x_sorted_boxes[j]['box'], min_y_overlap_ratio):
                line.append(j)
                skip_idxs.add(j)
                rightmost_box_idx = j

        # split line into lines if the distance between two neighboring
        # sub-lines' is greater than max_x_dist
        lines = []
        line_idx = 0
        lines.append([line[0]])
        rightmost = np.max(x_sorted_boxes[line[0]]['box'][::2])
        for k in range(1, len(line)):
            curr_box = x_sorted_boxes[line[k]]
            dist = np.min(curr_box['box'][::2]) - rightmost
            if dist > max_x_dist:
                line_idx += 1
                lines.append([])
            lines[line_idx].append(line[k])
            rightmost = max(rightmost, np.max(curr_box['box'][::2]))

        # Get merged boxes
        for box_group in lines:
            merged_box = {}
            merged_box['text'] = ' '.join(
                [x_sorted_boxes[idx]['text'] for idx in box_group])
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')
            for idx in box_group:
                x_max = max(np.max(x_sorted_boxes[idx]['box'][::2]), x_max)
                x_min = min(np.min(x_sorted_boxes[idx]['box'][::2]), x_min)
                y_max = max(np.max(x_sorted_boxes[idx]['box'][1::2]), y_max)
                y_min = min(np.min(x_sorted_boxes[idx]['box'][1::2]), y_min)
            merged_box['box'] = [
                x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max
            ]
            merged_boxes.append(merged_box)

    return merged_boxes


def is_seq_of(seq: Any,
              expected_type: Union[Type, tuple],
              seq_type: Type = None) -> bool:
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def crop_img(src_img, box, long_edge_pad_ratio=0.4, short_edge_pad_ratio=0.2):
    """Crop text region given the bounding box which might be slightly padded.
    The bounding box is assumed to be a quadrangle and tightly bound the text
    region.

    Args:
        src_img (np.array): The original image.
        box (list[float | int]): Points of quadrangle.
        long_edge_pad_ratio (float): The ratio of padding to the long edge. The
            padding will be the length of the short edge * long_edge_pad_ratio.
            Defaults to 0.4.
        short_edge_pad_ratio (float): The ratio of padding to the short edge.
            The padding will be the length of the long edge *
            short_edge_pad_ratio. Defaults to 0.2.

    Returns:
        np.array: The cropped image.
    """
    assert is_seq_of(box, (float, int))
    assert len(box) == 8
    assert 0. <= long_edge_pad_ratio < 1.0
    assert 0. <= short_edge_pad_ratio < 1.0

    h, w = src_img.shape[:2]
    points_x = np.clip(np.array(box[0::2]), 0, w)
    points_y = np.clip(np.array(box[1::2]), 0, h)

    box_width = np.max(points_x) - np.min(points_x)
    box_height = np.max(points_y) - np.min(points_y)
    shorter_size = min(box_height, box_width)

    if box_height < box_width:
        horizontal_pad = long_edge_pad_ratio * shorter_size
        vertical_pad = short_edge_pad_ratio * shorter_size
    else:
        horizontal_pad = short_edge_pad_ratio * shorter_size
        vertical_pad = long_edge_pad_ratio * shorter_size

    left = np.clip(int(np.min(points_x) - horizontal_pad), 0, w)
    top = np.clip(int(np.min(points_y) - vertical_pad), 0, h)
    right = np.clip(int(np.max(points_x) + horizontal_pad), 0, w)
    bottom = np.clip(int(np.max(points_y) + vertical_pad), 0, h)

    dst_img = src_img[top:bottom, left:right]

    return dst_img


import io
import os.path as osp
import warnings
from pathlib import Path
from typing import Optional, Union

import cv2
import mmengine.fileio as fileio
import numpy as np
from cv2 import (IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_IGNORE_ORIENTATION,
                 IMREAD_UNCHANGED)
from mmengine.utils import is_filepath, is_str

try:
    from turbojpeg import TJCS_RGB, TJPF_BGR, TJPF_GRAY, TurboJPEG
except ImportError:
    TJCS_RGB = TJPF_GRAY = TJPF_BGR = TurboJPEG = None

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None

try:
    import tifffile
except ImportError:
    tifffile = None

imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED,
    'color_ignore_orientation': IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    'grayscale_ignore_orientation':
        IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE
}

imread_backend = 'cv2'
supported_backends = ['cv2', 'turbojpeg', 'pillow', 'tifffile']


def _jpegflag(flag: str = 'color', channel_order: str = 'bgr'):
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'color':
        if channel_order == 'bgr':
            return TJPF_BGR
        elif channel_order == 'rgb':
            return TJCS_RGB
    elif flag == 'grayscale':
        return TJPF_GRAY
    else:
        raise ValueError('flag must be "color" or "grayscale"')


def _pillow2array(img,
                  flag: str = 'color',
                  channel_order: str = 'bgr') -> np.ndarray:
    """Convert a pillow image to numpy array.

    Args:
        img (:obj:`PIL.Image.Image`): The image loaded using PIL
        flag (str): Flags specifying the color type of a loaded image,
            candidates are 'color', 'grayscale' and 'unchanged'.
            Default to 'color'.
        channel_order (str): The channel order of the output image array,
            candidates are 'bgr' and 'rgb'. Default to 'bgr'.

    Returns:
        np.ndarray: The converted numpy array
    """
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'unchanged':
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # Handle exif orientation tag
        if flag in ['color', 'grayscale']:
            img = ImageOps.exif_transpose(img)
        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if img.mode != 'RGB':
            if img.mode != 'LA':
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert('RGB')
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert('RGBA')
                img = Image.new('RGB', img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        if flag in ['color', 'color_ignore_orientation']:
            array = np.array(img)
            if channel_order != 'rgb':
                array = array[:, :, ::-1]  # RGB to BGR
        elif flag in ['grayscale', 'grayscale_ignore_orientation']:
            img = img.convert('L')
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale", "unchanged", '
                f'"color_ignore_orientation" or "grayscale_ignore_orientation"'
                f' but got {flag}')
    return array


def imfrombytes(content: bytes,
                flag: str = 'color',
                channel_order: str = 'bgr',
                backend: Optional[str] = None) -> np.ndarray:
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Same as :func:`imread`.
        channel_order (str): The channel order of the output, candidates
            are 'bgr' and 'rgb'. Default to 'bgr'.
        backend (str | None): The image decoding backend type. Options are
            `cv2`, `pillow`, `turbojpeg`, `tifffile`, `None`. If backend is
            None, the global imread_backend specified by ``mmcv.use_backend()``
            will be used. Default: None.

    Returns:
        ndarray: Loaded image array.

    Examples:
        >>> img_path = '/path/to/img.jpg'
        >>> with open(img_path, 'rb') as f:
        >>>     img_buff = f.read()
        >>> img = mmcv.imfrombytes(img_buff)
        >>> img = mmcv.imfrombytes(img_buff, flag='color', channel_order='rgb')
        >>> img = mmcv.imfrombytes(img_buff, backend='pillow')
        >>> img = mmcv.imfrombytes(img_buff, backend='cv2')
    """

    if backend is None:
        backend = imread_backend
    if backend not in supported_backends:
        raise ValueError(
            f'backend: {backend} is not supported. Supported '
            "backends are 'cv2', 'turbojpeg', 'pillow', 'tifffile'")
    if backend == 'turbojpeg':
        img = jpeg.decode(  # type: ignore
            content, _jpegflag(flag, channel_order))
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        return img
    elif backend == 'pillow':
        with io.BytesIO(content) as buff:
            img = Image.open(buff)
            img = _pillow2array(img, flag, channel_order)
        return img
    elif backend == 'tifffile':
        with io.BytesIO(content) as buff:
            img = tifffile.imread(buff)
        return img
    else:
        img_np = np.frombuffer(content, np.uint8)
        flag = imread_flags[flag] if is_str(flag) else flag
        img = cv2.imdecode(img_np, flag)
        if flag == IMREAD_COLOR and channel_order == 'rgb':
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return img

