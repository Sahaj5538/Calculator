# src/vision/hand_detector.py

import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])

try:
    import cv2
except ImportError:
    print("Installing opencv-python...")
    install('opencv-python')
    import cv2

try:
    import mediapipe as mp
except ImportError:
    print("Installing mediapipe...")
    install('mediapipe')
    import mediapipe as mp


class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.7):
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

        # Finger tip ids
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        frame, handLms, self.mpHands.HAND_CONNECTIONS
                    )

        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        lmList = []

        if hasattr(self, "results") and self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[handNo]

                for id, lm in enumerate(hand.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((id, cx, cy))

                    if draw:
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return lmList

    # 🔥 NEW: Finger detection
    def fingersUp(self, lmList):
        fingers = []

        if len(lmList) == 0:
            return []

        # Thumb (special case)
        if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers
        for id in range(1, 5):
            if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers
