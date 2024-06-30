import cv2
import mediapipe as mp
import numpy as np
import random
import tensorflow as tf
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)


class HandTracker:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.8, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def getPostion(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def getUpFingers(self):
        fingers = []
        if self.lmList:
            # Thumb
            if self.lmList[4][1] < self.lmList[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # Fingers
            for id in range(8, 21, 4):
                if self.lmList[id][2] < self.lmList[id - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers


class ColorRect:
    def __init__(self, x, y, w, h, color, text='', alpha=0.5):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.text = text
        self.alpha = alpha

    def drawRect(self, img, text_color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):
        alpha = self.alpha
        bg_rec = img[self.y: self.y + self.h, self.x: self.x + self.w]
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8)
        white_rect[:] = self.color
        res = cv2.addWeighted(bg_rec, alpha, white_rect, 1 - alpha, 1.0)
        img[self.y: self.y + self.h, self.x: self.x + self.w] = res

        text_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)
        text_pos = (int(self.x + self.w / 2 - text_size[0][0] / 2), int(self.y + self.h / 2 + text_size[0][1] / 2))
        cv2.putText(img, self.text, text_pos, fontFace, fontScale, text_color, thickness)

    def isOver(self, x, y):
        return self.x + self.w > x > self.x and self.y + self.h > y > self.y


# Initialize the hand detector
detector = HandTracker(detectionCon=0.8)

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Create canvas to draw on
canvas = np.zeros((720, 1280, 3), np.uint8)

# Define a previous point to be used for drawing a line
px, py = 0, 0
# Initial brush color
color = (255, 0, 0)
brushSize = 5
eraserSize = 20

# Creating colors
colorsBtn = ColorRect(200, 0, 100, 100, (120, 255, 0), 'Colors')

colors = []
b = int(random.random() * 255)
g = int(random.random() * 255)
r = int(random.random() * 255)
colors.append(ColorRect(300, 0, 100, 100, (b, g, r)))
colors.append(ColorRect(400, 0, 100, 100, (0, 0, 255)))
colors.append(ColorRect(500, 0, 100, 100, (255, 0, 0)))
colors.append(ColorRect(600, 0, 100, 100, (0, 255, 0)))
colors.append(ColorRect(700, 0, 100, 100, (0, 255, 255)))
colors.append(ColorRect(800, 0, 100, 100, (0, 0, 0), "Eraser"))
clear = ColorRect(900, 0, 100, 100, (100, 100, 100), "Clear")

# Pen sizes
pens = [ColorRect(1100, 50 + 100 * i, 100, 100, (50, 50, 50), str(penSize)) for i, penSize in
        enumerate(range(5, 25, 5))]
penBtn = ColorRect(1100, 0, 100, 50, color, 'Pen')

# Whiteboard button
boardBtn = ColorRect(50, 0, 100, 100, (255, 255, 0), 'Board')
whiteBoard = ColorRect(50, 120, 1020, 580, (255, 255, 255), alpha=0.6)

coolingCounter = 20
hideBoard = True
hideColors = True
hidePenSizes = True

while True:
    if coolingCounter:
        coolingCounter -= 1

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.flip(frame, 1)

    frame = detector.findHands(frame)
    positions = detector.getPostion(frame, draw=False)
    upFingers = detector.getUpFingers()

    if upFingers:
        x, y = positions[8][1], positions[8][2]
        if upFingers[1] and not whiteBoard.isOver(x, y):
            px, py = 0, 0

            if not hidePenSizes:
                for pen in pens:
                    if pen.isOver(x, y):
                        brushSize = int(pen.text)
                        pen.alpha = 0
                    else:
                        pen.alpha = 0.5

            if not hideColors:
                for cb in colors:
                    if cb.isOver(x, y):
                        color = cb.color
                        cb.alpha = 0
                    else:
                        cb.alpha = 0.5

                if clear.isOver(x, y):
                    clear.alpha = 0
                    canvas = np.zeros((720, 1280, 3), np.uint8)
                else:
                    clear.alpha = 0.5

            if colorsBtn.isOver(x, y) and not coolingCounter:
                coolingCounter = 10
                colorsBtn.alpha = 0
                hideColors = not hideColors
                colorsBtn.text = 'Colors' if hideColors else 'Hide'
            else:
                colorsBtn.alpha = 0.5

            if penBtn.isOver(x, y) and not coolingCounter:
                coolingCounter = 10
                penBtn.alpha = 0
                hidePenSizes = not hidePenSizes
                penBtn.text = 'Pen' if hidePenSizes else 'Hide'
            else:
                penBtn.alpha = 0.5

            if boardBtn.isOver(x, y) and not coolingCounter:
                coolingCounter = 10
                boardBtn.alpha = 0
                hideBoard = not hideBoard
                boardBtn.text = 'Board' if hideBoard else 'Hide'
            else:
                boardBtn.alpha = 0.5
        elif upFingers[1] and not upFingers[2]:
            if whiteBoard.isOver(x, y) and not hideBoard:
                cv2.circle(frame, (x, y), brushSize, color, -1)
                if px == 0 and py == 0:
                    px, py = x, y
                if color == (0, 0, 0):
                    cv2.line(canvas, (px, py), (x, y), color, eraserSize)
                else:
                    cv2.line(canvas, (px, py), (x, y), color, brushSize)
                px, py = x, y
        else:
            px, py = 0, 0

    colorsBtn.drawRect(frame)
    cv2.rectangle(frame, (colorsBtn.x, colorsBtn.y), (colorsBtn.x + colorsBtn.w, colorsBtn.y + colorsBtn.h),
                  (255, 255, 255), 2)

    boardBtn.drawRect(frame)
    cv2.rectangle(frame, (boardBtn.x, boardBtn.y), (boardBtn.x + boardBtn.w, boardBtn.y + boardBtn.h), (255, 255, 255),
                  2)

    if not hideBoard:
        whiteBoard.drawRect(frame)
        canvasGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(canvasGray, 20, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, imgInv)
        frame = cv2.bitwise_or(frame, canvas)

    if not hideColors:
        for c in colors:
            c.drawRect(frame)
            cv2.rectangle(frame, (c.x, c.y), (c.x + c.w, c.y + c.h), (255, 255, 255), 2)
        clear.drawRect(frame)
        cv2.rectangle(frame, (clear.x, clear.y), (clear.x + clear.w, clear.y + clear.h), (255, 255, 255), 2)

    penBtn.color = color
    penBtn.drawRect(frame)
    cv2.rectangle(frame, (penBtn.x, penBtn.y), (penBtn.x + penBtn.w, penBtn.y + penBtn.h), (255, 255, 255), 2)
    if not hidePenSizes:
        for pen in pens:
            pen.drawRect(frame)
            cv2.rectangle(frame, (pen.x, pen.y), (pen.x + pen.w, pen.y + pen.h), (255, 255, 255), 2)

    cv2.imshow('video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
