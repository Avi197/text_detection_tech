import mmengine
from mmcv import imfrombytes
from mmocr.apis import MMOCRInferencer

from text_detection_mmocr.utils import bbox2poly, poly2bbox, crop_img

import cv2
import imutils

import numpy as np
from PIL import Image

from common_utils.img_utils import four_point_transform
from scipy.spatial import distance as dist


# from pipeline.data_obj.ann import TextAnn
# from pipeline.base_obj import BaseDetector


def find_largest_polygon(polygons):
    polygons = [np.array(i).astype(int) for i in polygons]
    # Find the polygon with the largest area
    largest_area = 0
    largest_polygon = None
    for polygon in polygons:
        area = cv2.contourArea(polygon)
        if area > largest_area:
            largest_area = area
            largest_polygon = polygon
    return largest_polygon


def get_rotate_angle(rect):
    # Rotate the image using the rotate function
    rect = cv2.minAreaRect(rect)
    if rect[1][0] < rect[1][1]:
        angle = -(rect[2] - 90)
    else:
        angle = -rect[2]
    return angle


def convert_to_ocr_line(coord, img, add_margin=0.2, model_height=64):
    h, w = img.shape

    x_min = int(min(coord[::2]))
    x_max = int(max(coord[::2]))
    y_min = int(min(coord[1::2]))
    y_max = int(max(coord[1::2]))

    margin = int(add_margin * (y_max - y_min))

    x_min = max(0, x_min - margin)
    x_max = min(w, x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(h, y_max + margin)

    crop_img = img[y_min: y_max, x_min:x_max]

    ratio = (x_max - x_min) / (y_max - y_min)
    crop_img = cv2.resize(crop_img, (int(model_height * ratio), model_height),
                          interpolation=Image.ANTIALIAS)
    update_coord = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    return TextAnn.from_coord_and_img(update_coord, w, h, crop_img)


def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")


class MMOCRTextDetection:
    def __init__(self, det_model_dir='', device=False):
        if device == 'cuda':
            device = 'cuda:0'
        else:
            device = None
        self.text_detector = MMOCRInferencer(det=det_model_dir, device=device)

    def predict(self, img):
        result_mmocr = self.text_detector(img)
        result = []

        # img_bytes = mmengine.fileio.get(img)
        # img_mmocr = imfrombytes(img_bytes)

        for polygon in result_mmocr['predictions'][0]['det_polygons']:
            quad = bbox2poly(poly2bbox(polygon)).tolist()
            result.append(quad)
            # result.append(crop_img(img_mmocr, quad))

        largest_polygon = find_largest_polygon(result)
        angle = get_rotate_angle(largest_polygon)
        if int(angle) != 0:
            img = imutils.rotate_bound(img, angle)
            result = self.text_detector(img)

            # result = self.text_detector.ocr(img=img, cls=self.cls, det=self.det, rec=self.rec)

        img_cv_grey = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

        img_list = []
        for r in result:
            coord = [i for sub in r for i in sub]
            ocr_line = convert_to_ocr_line(coord, img_cv_grey)
            img_list.append(ocr_line)
        return img_list


def convert_poly_to_ocr_line(poly, img, add_margin=0.2, model_height=64):
    h, w = img.shape

    (tl, tr, br, bl) = poly
    tl_bl_dist = np.linalg.norm(np.array(tl) - np.array(bl))
    tr_br_dist = np.linalg.norm(np.array(tr) - np.array(br))
    margin = int(add_margin * min(tl_bl_dist, tr_br_dist))

    padded_poly = [
        [max(0, tl[0] - margin), max(0, tl[1] - margin)],
        [min(w, tr[0] + margin), max(0, tr[1] - margin)],
        [min(w, br[0] + margin), min(h, br[1] + margin)],
        [max(0, bl[0] - margin), min(h, bl[1] + margin)]
    ]

    padded_rect = np.array(padded_poly, dtype="float32")
    cropped_img = four_point_transform(img, padded_rect)

    ratio = cropped_img.shape[1] / cropped_img.shape[0]
    crop_img = cv2.resize(cropped_img, (int(model_height * ratio), model_height),
                          interpolation=Image.ANTIALIAS)
    return TextAnn.from_coord_and_img(poly, w, h, crop_img)


class MMOCRTextDetectionPoly(MMOCRTextDetection):
    def predict(self, img):
        result_mmocr = self.text_detector(img)
        result = []
        for polygon in result_mmocr['predictions'][0]['det_polygons']:
            it = iter(polygon)
            quad = cv2.minAreaRect(np.array([*zip(it, it)]).astype(int))
            box = cv2.boxPoints(quad)
            box = np.int0(box)
            box = order_points(box)
            result.append(box)
        img_cv_grey = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

        img_list = []
        polys = sorted(result, key=lambda x: x[0][1])

        for poly in polys:
            ocr_line = convert_poly_to_ocr_line(poly, img_cv_grey)
            img_list.append(ocr_line)
        return img_list
