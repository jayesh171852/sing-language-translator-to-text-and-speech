"""
data_collection_final.py — Collect hand skeleton images for training.

Controls:
  'a' — toggle auto-capture (captures 180 images per letter)
  'n' — move to next letter
  ESC — quit
"""
import cv2
import os
import traceback
import numpy as np
from hand_detector import HandDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHITE_PATH = os.path.join(BASE_DIR, 'white.jpg')

# Output dataset directory — change this to wherever you want to save data
DATASET_DIR = os.path.join(BASE_DIR, 'AtoZ_3.1')

# Generate white image if missing
if not os.path.exists(WHITE_PATH):
    white_img = np.ones((400, 400, 3), np.uint8) * 255
    cv2.imwrite(WHITE_PATH, white_img)

capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

c_dir = 'A'
letter_dir = os.path.join(DATASET_DIR, c_dir)
os.makedirs(letter_dir, exist_ok=True)
count = len(os.listdir(letter_dir))

offset = 15
step = 1
flag = False
suv = 0

while True:
    try:
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        white = cv2.imread(WHITE_PATH)

        hands, _ = hd.findHands(frame, draw=False, flipType=True)

        skeleton1 = None

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            y1 = max(0, y - offset)
            y2 = min(frame.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(frame.shape[1], x + w + offset)
            image = np.array(frame[y1:y2, x1:x2])

            if image.size > 0:
                handz, _ = hd2.findHands(image, draw=True, flipType=True)
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

                    skeleton1 = np.array(white)
                    cv2.imshow("Skeleton", skeleton1)

        label = f"Letter: {c_dir}  Count: {count}  {'[CAPTURING]' if flag else ''}"
        frame = cv2.putText(frame, label, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0) if flag else (255, 0, 0), 2)
        cv2.imshow("frame", frame)

        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:
            break

        if interrupt & 0xFF == ord('n'):
            c_dir = chr(ord(c_dir) + 1)
            if ord(c_dir) > ord('Z'):
                c_dir = 'A'
            flag = False
            letter_dir = os.path.join(DATASET_DIR, c_dir)
            os.makedirs(letter_dir, exist_ok=True)
            count = len(os.listdir(letter_dir))
            print(f"Switched to letter: {c_dir}, existing count: {count}")

        if interrupt & 0xFF == ord('a'):
            flag = not flag
            if flag:
                suv = 0
                print(f"Started capturing for: {c_dir}")
            else:
                print("Capture paused.")

        if flag and skeleton1 is not None:
            if suv >= 180:
                flag = False
                print(f"Done capturing {c_dir}")
            elif step % 3 == 0:
                save_path = os.path.join(DATASET_DIR, c_dir, f"{count}.jpg")
                cv2.imwrite(save_path, skeleton1)
                count += 1
                suv += 1
                print(f"Saved {save_path} ({suv}/180)")
            step += 1

    except Exception:
        print("Error:", traceback.format_exc())

capture.release()
cv2.destroyAllWindows()
