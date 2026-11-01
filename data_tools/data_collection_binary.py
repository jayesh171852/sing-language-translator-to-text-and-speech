"""
data_collection_binary.py — Collect binary/gray hand images for training.

Controls:
  'a' — toggle auto-capture
  'n' — next letter
  ESC — quit
"""
import cv2
import os
import traceback
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
import sys
for p in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'src')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from hand_detector import HandDetector

try:
    import config
    WHITE_PATH = config.WHITE_PATH
    BINARY_DIR = os.path.join(config.DATA_DIR, 'binary_imgs')
    GRAY_DIR = os.path.join(config.DATA_DIR, 'gray_imgs')
except Exception:
    WHITE_PATH = os.path.join(PROJECT_ROOT, 'assets', 'white.jpg')
    BINARY_DIR = os.path.join(PROJECT_ROOT, 'data', 'binary_imgs')
    GRAY_DIR = os.path.join(PROJECT_ROOT, 'data', 'gray_imgs')

# Generate white image if missing
if not os.path.exists(WHITE_PATH):
    white_img = np.ones((400, 400, 3), np.uint8) * 255
    cv2.imwrite(WHITE_PATH, white_img)

capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

p_dir = 'A'
for d in [BINARY_DIR, GRAY_DIR]:
    os.makedirs(os.path.join(d, p_dir), exist_ok=True)
count = len(os.listdir(os.path.join(GRAY_DIR, p_dir)))

offset = 30
step = 1
flag = False
suv = 0

while True:
    try:
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        white = cv2.imread(WHITE_PATH)

        hands, _ = hd.findHands(frame, draw=False, flipType=True)

        img_final = img_final1 = None

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            y1 = max(0, y - offset)
            y2 = min(frame.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(frame.shape[1], x + w + offset)
            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                # Gray image
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (1, 1), 2)

                # Binary image
                gray2 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur2 = cv2.GaussianBlur(gray2, (5, 5), 2)
                th3 = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
                _, test_image = cv2.threshold(th3, 27, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                img_final1 = np.ones((400, 400), np.uint8) * 148
                h_r, w_r = blur.shape
                img_final1[((400-h_r)//2):((400-h_r)//2)+h_r,
                           ((400-w_r)//2):((400-w_r)//2)+w_r] = blur

                img_final = np.ones((400, 400), np.uint8) * 255
                h_r, w_r = test_image.shape
                img_final[((400-h_r)//2):((400-h_r)//2)+h_r,
                          ((400-w_r)//2):((400-w_r)//2)+w_r] = test_image

                # Skeleton
                handz, _ = hd2.findHands(roi, draw=False, flipType=True)
                if handz:
                    hand2 = handz[0]
                    pts = hand2['lmList']
                    os_x = ((400 - w) // 2) - 15
                    os_y = ((400 - h) // 2) - 15

                    for t in list(range(0,4)) + list(range(5,8)) + list(range(9,12)) + list(range(13,16)) + list(range(17,20)):
                        cv2.line(white,
                                 (pts[t][0]+os_x, pts[t][1]+os_y),
                                 (pts[t+1][0]+os_x, pts[t+1][1]+os_y),
                                 (0, 255, 0), 3)
                    for a, b in [(5,9),(9,13),(13,17),(0,5),(0,17)]:
                        cv2.line(white,
                                 (pts[a][0]+os_x, pts[a][1]+os_y),
                                 (pts[b][0]+os_x, pts[b][1]+os_y),
                                 (0, 255, 0), 3)
                    for i in range(21):
                        cv2.circle(white, (pts[i][0]+os_x, pts[i][1]+os_y), 2, (0, 0, 255), 1)

                    cv2.imshow("Skeleton", white)

                cv2.imshow("Binary", img_final)

        label = f"Letter: {p_dir}  Count: {count}  {'[CAPTURING]' if flag else ''}"
        frame = cv2.putText(frame, label, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0) if flag else (255, 0, 0), 2)
        cv2.imshow("frame", frame)

        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:
            break

        if interrupt & 0xFF == ord('n'):
            p_dir = chr(ord(p_dir) + 1)
            if ord(p_dir) > ord('Z'):
                p_dir = 'A'
            flag = False
            for d in [BINARY_DIR, GRAY_DIR]:
                os.makedirs(os.path.join(d, p_dir), exist_ok=True)
            count = len(os.listdir(os.path.join(GRAY_DIR, p_dir)))
            print(f"Switched to: {p_dir}, count: {count}")

        if interrupt & 0xFF == ord('a'):
            flag = not flag
            if flag:
                suv = 0

        if flag and img_final is not None and img_final1 is not None:
            if suv >= 50:
                flag = False
            elif step % 2 == 0:
                cv2.imwrite(os.path.join(BINARY_DIR, p_dir, f"{p_dir.lower()}{count}.jpg"), img_final)
                cv2.imwrite(os.path.join(GRAY_DIR, p_dir, f"{p_dir.lower()}{count}.jpg"), img_final1)
                count += 1
                suv += 1
                print(f"Saved {p_dir} ({suv}/50)")
            step += 1

    except Exception:
        print("Error:", traceback.format_exc())

capture.release()
cv2.destroyAllWindows()
