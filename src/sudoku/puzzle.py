from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2


def find_puzzle(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    treshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Visualizing each step of the image processing pipeline
    if debug:
        cv2.imshow('Puzzle Treshold', treshold)
        cv2.waitKey(0)
    
    # Find contours in the tresholded image and sort them by size in descending order
    contours = cv2.findContours(
        tresh.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    puzzleContour = None

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour, 
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
