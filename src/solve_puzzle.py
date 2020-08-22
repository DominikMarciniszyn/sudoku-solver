from sudoku.puzzle import (
    find_puzzle,
    extract_digit
)
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from Sudoku import Sudoku
import numpy as np
import argparse
import imutils
import cv2


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument(
    '-m',
    '--model',
    required=True,
    help='Path to trained digit classifier'
)
argument_parser.add_argument(
    '-i',
    '--image',
    required=True,
    help='Path to input Sudoku puzzle image'
)
argument_parser.add_argument(
    '-d',
    '--debug',
    type=int,
    default=-1,
    help='Whether or not we are visualizing each step of the pipeline'
)
args = vars(argument_parser.parse_args())

print('Loading digit classifier...')
model = load_model(args['model'])

print('Processing image...')
image = cv2.imread(args['image'])
image = imutils.resize(image, width=600)

(puzzleImage, wrapped) = find_puzzle(image, debug=args['debug'] > 0)

board = np.zeros((9, 9), dtype='int')

stepX = wrapped.shape[1]
stepY = wrapped.shape[0]

cellLocations = []

for y in range(0, 9):
    row = []

    for x in range(0, 9):
        startX = x * stepX
        startY = y * stepY
        endX = (x + 1) * stepX
        endY = (y + 1) * stepY

        row.append((startX, startY, endX, endY))

        cell = wrapped[startY:endY, startX:endX]
        digit = extract_digit(cell, debug=args['debug'] > 0)

        if digit is not None:
            roi = cv2.resize(digit, (28, 28))
            roi = roi.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi).argmax(axis=1)[0]
            board[y, x] = prediction

    cellLocations.append(row)

print('Sudoku board:')
puzzle = Sudoku(3, 3, board=board.tolist())
puzzle.show()

print('Solving Sudoku puzzle...')
solution = puzzle.solve()
solution.show_full()


# Visualize the solution
for (cellRow, boardRow) in zip(cellLocations, solution.board):
    for (box, digit) in zip(cellRow, boardRow):
        startX, startY, endX, endY = box
        textX = int((endX - startX) * 0.33)
        textY = int((endY - startY) * -0.2)
        textX += startX
        textY += endY

        cv2.putText(
            puzzleImage,
            str(digit),
            (textX, textY),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

cv2.imshow('Sudoku Result', puzzleImage)
cv2.waitKey(0)
