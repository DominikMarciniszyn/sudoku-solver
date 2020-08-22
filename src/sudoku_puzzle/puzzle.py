from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2


def find_puzzle(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    threshold = cv2.bitwise_not(threshold)

    # Visualizing each step of the image processing pipeline
    if debug:
        cv2.imshow('Puzzle Treshold', threshold)
        cv2.waitKey(0)
    
    # Find contours in the thresh image and sort them by size in descending order
    contours = cv2.findContours(
        threshold.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    puzzleContour = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(
            c, 
            0.02 * peri,
            True
        )

        if len(approx) == 4:
            puzzleContour = approx
            break
    
    if puzzleContour is None:
        raise Exception('Could not find Sudoku puzzle...')

    if debug:
        output = image.copy()
        cv2.drawContours(output, [puzzleContour], -1, (0, 255, 0), 2)
        cv2.imshow('Puzzle', output)
        cv2.waitKey(0)

    puzzle = four_point_transform(image, puzzleContour.reshape(4, 2))
    wrapped = four_point_transform(gray, puzzleContour.reshape(4, 2))

    if debug:
        cv2.imshow('Puzzle', output)
        cv2.waitKey(0)
    
    return (puzzle, wrapped)


def extract_digit(cell, debug=False):
    threshold = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    threshold = clear_border(threshold)

    if debug:
        cv2.imshow('Cell Thresh', threshold)
        cv2.waitKey(0)

    contour = cv2.findContours(
        threshold.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contour = imutils.grab_contours(contour)

    if len(contour) == 0:
        return None

    c = max(contour, key=cv2.contourArea)
    mask = np.zeros(threshold.shape, dtype='uint8')
    cv2.drawContours(mask, [c], -1, 255, -1)

    (height, width) = threshold.shape
    percentFilled = cv2.countNonZero(mask) / float(width * height)

    if percentFilled < 0.03:
        return None

    digit = cv2.bitwise_and(threshold, threshold, mask=mask)

    if debug:
        cv2.imshow('Cell Thresh', threshold)
        cv2.waitKey(0)

    return digit
