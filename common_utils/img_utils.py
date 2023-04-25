import cv2
import numpy as np


def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


def intersect(rect1, rect2):
    """
    Checks if two rectangles intersect.

    Parameters:
    rect1 (tuple): (x1, y1, x2, y2) coordinates of the first rectangle
    rect2 (tuple): (x1, y1, x2, y2) coordinates of the second rectangle

    Returns:
    bool: True if the rectangles intersect, False otherwise
    """
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    if (x1 > x4) or (x2 < x3) or (y1 > y4) or (y2 < y3):
        return False
    else:
        return True
