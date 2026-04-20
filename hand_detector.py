"""
hand_detector.py — HandDetector using mediapipe Tasks API (0.10.30+).
Mimics the cvzone HandDetector interface used in this project.
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'hand_landmarker.task')


class HandDetector:
    def __init__(self, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        self.maxHands = maxHands

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=maxHands,
            min_hand_detection_confidence=detectionCon,
            min_hand_presence_confidence=detectionCon,
            min_tracking_confidence=minTrackCon,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

    def findHands(self, img, draw=True, flipType=True):
        """
        Returns (hand_list, img).
        Each hand dict: {'bbox': (x,y,w,h), 'lmList': [[x,y,z],...], 'type': str}
        """
        h, w = img.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result = self.detector.detect(mp_image)

        hand_list = []

        for i, hand_lms in enumerate(result.hand_landmarks):
            lm_list = []
            x_list, y_list = [], []

            for lm in hand_lms:
                px = int(lm.x * w)
                py = int(lm.y * h)
                pz = int(lm.z * w)
                lm_list.append([px, py, pz])
                x_list.append(px)
                y_list.append(py)

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            bbox = (xmin, ymin, xmax - xmin, ymax - ymin)

            hand_type = "Right"
            if result.handedness and i < len(result.handedness):
                label = result.handedness[i][0].category_name
                hand_type = ("Left" if label == "Right" else "Right") if flipType else label

            hand_list.append({
                'lmList': lm_list,
                'bbox': bbox,
                'type': hand_type,
            })

            if draw:
                # Draw landmarks manually
                connections = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),
                    (9,13),(13,14),(14,15),(15,16),
                    (13,17),(17,18),(18,19),(19,20),
                    (0,17),
                ]
                for a, b in connections:
                    cv2.line(img, tuple(lm_list[a][:2]), tuple(lm_list[b][:2]), (0, 255, 0), 2)
                for pt in lm_list:
                    cv2.circle(img, tuple(pt[:2]), 4, (0, 0, 255), -1)

        return hand_list, img
