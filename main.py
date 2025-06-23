import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import sys
print("Python path:", sys.executable)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# MediaPipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Pycaw volume setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

vol = 0
volBar = 400
volPer = 0

# 🔧 Tweak here: reduced range for easier control
minGesture = 20   # Less than this = mute
maxGesture = 150  # More than this = full volume

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        if lmList:
            x1, y1 = lmList[4][1], lmList[4][2]   # Thumb tip
            x2, y2 = lmList[8][1], lmList[8][2]   # Index tip

            # Draw visuals
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), -1)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), -1)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Calculate distance
            length = math.hypot(x2 - x1, y2 - y1)

            # Display distance on screen
            cv2.putText(img, f'Dist: {int(length)} px', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            # Interpolate to volume range
            vol = np.interp(length, [minGesture, maxGesture], [minVol, maxVol])
            volBar = np.interp(length, [minGesture, maxGesture], [400, 150])
            volPer = np.interp(length, [minGesture, maxGesture], [0, 100])

            # Set volume
            volume.SetMasterVolumeLevel(vol, None)

            # Visual feedback when muted
            if length < minGesture + 5:
                cv2.circle(img, ((x1 + x2)//2, (y1 + y2)//2), 12, (0, 255, 0), -1)

    # Draw volume bar
    cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), -1)
    cv2.putText(img, f'{int(volPer)} %', (35, 430), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Display
    cv2.imshow("🖐️ Gesture Volume Control", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
