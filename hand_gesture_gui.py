



import cv2
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw
import math

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# Function to detect simple gestures
def detect_gesture(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    finger_mcp = [5, 9, 13, 17]     # Corresponding MCP joints
    thumb_tip = 4
    thumb_mcp = 2
    
    fingers_open = 0
    for tip, mcp in zip(finger_tips, finger_mcp):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:
            fingers_open += 1

    # Thumb
    if hand_landmarks.landmark[thumb_tip].x > hand_landmarks.landmark[thumb_mcp].x:
        fingers_open += 1

    # Determine gesture
    if fingers_open == 0:
        return "FIST"
    elif fingers_open == 5:
        return "PALM"
    elif fingers_open == 1 and hand_landmarks.landmark[thumb_tip].x > hand_landmarks.landmark[thumb_mcp].x:
        return "THUMBS UP"
    else:
        return "UNKNOWN"

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks)
            cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


