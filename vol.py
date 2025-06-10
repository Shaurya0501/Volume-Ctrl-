import cv2
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Setup Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]  # typically -65.25
max_vol = vol_range[1]  # typically 0.0

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get coordinates of thumb tip (landmark 4) and index finger tip (landmark 8)
        h, w, _ = img.shape
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]

        x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
        x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

        # Draw circles on thumb and index finger tips
        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Calculate distance between thumb and index finger
        length = math.hypot(x2 - x1, y2 - y1)

        # Map length (distance) to volume level
        # Tune these min and max length values based on your hand distance
        min_length, max_length = 30, 200

        # Clamp the length within min and max lengths
        length = max(min(length, max_length), min_length)

        # Interpolate volume from length
        vol = ((length - min_length) / (max_length - min_length)) * (max_vol - min_vol) + min_vol
        volume.SetMasterVolumeLevel(vol, None)

        # Display volume percentage
        vol_percent = int(((vol - min_vol) / (max_vol - min_vol)) * 100)
        cv2.putText(img, f'Vol: {vol_percent} %', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

        # Draw hand landmarks
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Volume Control", img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
