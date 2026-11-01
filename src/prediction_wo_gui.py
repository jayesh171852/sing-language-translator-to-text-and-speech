"""
prediction_wo_gui.py — Run sign language prediction without GUI (console + OpenCV windows).
Press ESC to quit.
"""
import math
import os
import sys
import traceback

import cv2
import numpy as np

# Ensure local src/ and project root are in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
for p in (BASE_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.hand_detector import HandDetector
except ImportError:
    from hand_detector import HandDetector
from tensorflow.keras.models import load_model

try:
    import config
    MODEL_PATH = config.CNN_MODEL_PATH
    WHITE_PATH = config.WHITE_PATH
except Exception:
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets')
    MODEL_PATH = os.path.join(MODELS_DIR, 'cnn8grps_rad1_model.h5') if os.path.exists(os.path.join(MODELS_DIR, 'cnn8grps_rad1_model.h5')) else os.path.join(BASE_DIR, 'cnn8grps_rad1_model.h5')
    WHITE_PATH = os.path.join(ASSETS_DIR, 'white.jpg') if os.path.exists(os.path.join(ASSETS_DIR, 'white.jpg')) else os.path.join(BASE_DIR, 'white.jpg')

# Generate white image if missing
if not os.path.exists(WHITE_PATH):
    os.makedirs(os.path.dirname(WHITE_PATH), exist_ok=True)
    white_img = np.ones((400, 400, 3), np.uint8) * 255
    cv2.imwrite(WHITE_PATH, white_img)

model = load_model(MODEL_PATH)
capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

offset = 29


def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))


while True:
    try:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        hands, _ = hd.findHands(frame, draw=False, flipType=True)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            y1 = max(0, y - offset)
            y2 = min(frame.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(frame.shape[1], x + w + offset)
            image = frame[y1:y2, x1:x2]

            if image.size == 0:
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            white = cv2.imread(WHITE_PATH)
            handz, _ = hd2.findHands(image, draw=False, flipType=True)

            if handz:
                hand2 = handz[0]
                pts = hand2['lmList']

                os_x = ((400 - w) // 2) - 15
                os_y = ((400 - h) // 2) - 15

                for t in list(range(0, 4)) + list(range(5, 8)) + list(range(9, 12)) + list(range(13, 16)) + list(range(17, 20)):
                    cv2.line(white,
                             (pts[t][0] + os_x, pts[t][1] + os_y),
                             (pts[t+1][0] + os_x, pts[t+1][1] + os_y),
                             (0, 255, 0), 3)

                for a, b in [(5,9),(9,13),(13,17),(0,5),(0,17)]:
                    cv2.line(white,
                             (pts[a][0] + os_x, pts[a][1] + os_y),
                             (pts[b][0] + os_x, pts[b][1] + os_y),
                             (0, 255, 0), 3)

                for i in range(21):
                    cv2.circle(white, (pts[i][0] + os_x, pts[i][1] + os_y), 2, (0, 0, 255), 1)

                cv2.imshow("Skeleton", white)

                white_input = white.reshape(1, 400, 400, 3)
                prob = np.array(model.predict(white_input, verbose=0)[0], dtype='float32')
                ch1 = np.argmax(prob)
                prob[ch1] = 0
                ch2 = np.argmax(prob)

                pl = [ch1, ch2]

                # Group disambiguation (same logic as final_pred.py)
                if pl in [[5,2],[5,3],[3,5],[3,6],[3,0],[3,2],[6,4],[6,1],[6,2],[6,6],[6,7],[6,0],[6,5],
                           [4,1],[1,0],[1,1],[6,3],[1,6],[5,6],[5,1],[4,5],[1,4],[1,5],[2,0],[2,6],[4,6],
                           [5,7],[7,6],[2,5],[7,1],[5,4],[7,0],[7,5],[7,2]]:
                    if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and
                            pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                        ch1 = 0

                pl = [ch1, ch2]
                if pl in [[0,0],[0,6],[0,2],[0,5],[0,1],[0,7],[5,2],[7,6],[7,1]]:
                    if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and
                            pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and
                            pts[0][0] > pts[20][0] and pts[5][0] > pts[4][0]):
                        ch1 = 2

                # Subgroup classification
                if ch1 == 0:
                    ch1 = 'S'
                    if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
                        ch1 = 'A'
                    if (pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and
                            pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]):
                        ch1 = 'T'
                    if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
                        ch1 = 'E'
                    if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]:
                        ch1 = 'M'
                    if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
                        ch1 = 'N'

                if ch1 == 2:
                    ch1 = 'C' if distance(pts[12], pts[4]) > 42 else 'O'
                if ch1 == 3:
                    ch1 = 'G' if distance(pts[8], pts[12]) > 72 else 'H'
                if ch1 == 7:
                    ch1 = 'Y' if distance(pts[8], pts[4]) > 42 else 'J'
                if ch1 == 4:
                    ch1 = 'L'
                if ch1 == 6:
                    ch1 = 'X'
                if ch1 == 5:
                    if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
                        ch1 = 'Z' if pts[8][1] < pts[5][1] else 'Q'
                    else:
                        ch1 = 'P'
                if ch1 == 1:
                    if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
                        ch1 = 'B'
                    elif pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]:
                        ch1 = 'D'
                    elif pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
                        ch1 = 'F'
                    elif pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]:
                        ch1 = 'I'
                    elif pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]:
                        ch1 = 'W'
                    elif (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and
                          pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[4][1] < pts[9][1]):
                        ch1 = 'K'
                    elif ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8 and
                          pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and
                          pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                        ch1 = 'U'
                    elif ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8 and
                          pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and
                          pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[4][1] > pts[9][1]):
                        ch1 = 'V'
                    elif (pts[8][0] > pts[12][0] and pts[6][1] > pts[8][1] and
                          pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                        ch1 = 'R'

                print(f"Predicted: {ch1}")
                frame = cv2.putText(frame, f"Predicted: {ch1}", (30, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception:
        print("Error:", traceback.format_exc())

capture.release()
cv2.destroyAllWindows()
