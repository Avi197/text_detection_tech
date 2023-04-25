import cv2
import imutils

import numpy as np
from PIL import Image

from common_utils.img_utils import four_point_transform
# from pipeline.data_obj.ann import TextAnn
from text_detector_paddleocr.CustomPaddleOCR import CustomPaddleOCR

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


class PaddleTextDetection:
    def __init__(self, det_model_dir='default_model',
                 use_angle_cls=True, device='cpu', cls=True,
                 det=True, rec=False, lang="en", add_margin=0.2):
        self.cls = cls
        self.det = det
        self.rec = rec
        self.lang = lang
        self.det_model_dir = det_model_dir
        use_gpu = False
        if device == 'cuda':
            use_gpu = True

        self.text_detector = CustomPaddleOCR(use_angle_cls=use_angle_cls, use_gpu=use_gpu,
                                             det_model_dir=self.det_model_dir,
                                             lang=self.lang)
        self.add_margin = add_margin

    def predict(self, img):
        result = self.text_detector.ocr(img=img, cls=self.cls, det=self.det, rec=self.rec)
        largest_polygon = find_largest_polygon(result[0])
        angle = get_rotate_angle(largest_polygon)
        if int(angle) != 0:
            img = imutils.rotate_bound(img, angle)
            result = self.text_detector.ocr(img=img, cls=self.cls, det=self.det, rec=self.rec)

        img_cv_grey = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

        img_list = []
        for r in result[0]:
            coord = [i for sub in r for i in sub]
            ocr_line = convert_to_ocr_line(coord, img_cv_grey)
            img_list.append(ocr_line)
        return img_list


def convert_poly_to_ocr_line(poly, img, add_margin=0.2, model_height=64):
    h, w = img.shape

    (tl, tr, br, bl) = poly

    # tl_bl_dist = np.linalg.norm(np.array(tl) - np.array(bl))
    # tr_br_dist = np.linalg.norm(np.array(tr) - np.array(br))
    # margin = int(add_margin * min(tl_bl_dist, tr_br_dist))

    margin_x = 10
    margin_y = 5

    padded_poly = [
        [max(0, tl[0] - margin_x), max(0, tl[1] - margin_y)],
        [min(w, tr[0] + margin_x), max(0, tr[1] - margin_y)],
        [min(w, br[0] + margin_x), min(h, br[1] + margin_y)],
        [max(0, bl[0] - margin_x), min(h, bl[1] + margin_y)]
    ]

    padded_rect = np.array(padded_poly, dtype="float32")
    cropped_img = four_point_transform(img, padded_rect)

    ratio = cropped_img.shape[1] / cropped_img.shape[0]
    crop_img = cv2.resize(cropped_img, (int(model_height * ratio), model_height),
                          interpolation=Image.ANTIALIAS)
    return TextAnn.from_coord_and_img(poly, w, h, crop_img)


class PaddleTextDetectionPoly(PaddleTextDetection):
    def predict(self, img):
        result = self.text_detector.ocr(img=img, cls=self.cls, det=self.det, rec=self.rec)
        img_cv_grey = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

        img_list = []
        polys = sorted(result[0], key=lambda x: x[0][1])
        for poly in polys:
            ocr_line = convert_poly_to_ocr_line(poly, img_cv_grey, add_margin=self.add_margin)
            img_list.append(ocr_line)
        return img_list
