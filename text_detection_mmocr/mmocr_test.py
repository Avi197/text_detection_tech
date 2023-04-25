import mmengine
from mmcv import imfrombytes
from mmocr.apis import MMOCRInferencer
import cv2
import imutils

import numpy as np
from PIL import Image
from pathlib import Path
from scipy.spatial import distance as dist
from common_utils.img_utils import four_point_transform
# from pipeline.data_obj.ann import TextAnn


from text_detection_mmocr.utils import bbox2poly, poly2bbox, crop_img

ocr = MMOCRInferencer(det='TextSnake', device='cuda:0')


def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")


input_folder = 'img_folder'
glob_pattern = ''
for input_file in Path(input_folder).glob(glob_pattern):
    result = ocr(input_file.as_posix(), show=True)
    result_polygon = []

    img_bytes = mmengine.fileio.get(input_file)
    img_mmocr = imfrombytes(img_bytes)

    for polygon in result['predictions'][0]['det_polygons']:
        it = iter(polygon)
        quad = cv2.minAreaRect(np.array([*zip(it, it)]).astype(int))
        box = cv2.boxPoints(quad)
        box = np.int0(box)
        box = order_points(box)
        result.append(box)
