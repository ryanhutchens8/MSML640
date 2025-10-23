import cv2
import numpy as np

video_path = r"Videos\Raw Video\GX010312_trimmed.mp4"
cap = cv2.VideoCapture(video_path)

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

SAT_THRESH  = 80   # minimal saturation
BLUE_THRESH = 10  # minimal blue channel intensity
scale = 0.5        # shrink display

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # saturation mask
    _, mask_s = cv2.threshold(s, SAT_THRESH, 255, cv2.THRESH_BINARY)

    # blue channel mask
    b, g, r = cv2.split(frame)
    _, mask_b = cv2.threshold(b, BLUE_THRESH, 255, cv2.THRESH_BINARY)

    # combine both (AND)
    mask = cv2.bitwise_and(mask_s, mask_b)

    # resize for viewing
    disp_frame = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
    disp_mask  = cv2.resize(mask,  (int(mask.shape[1]*scale), int(mask.shape[0]*scale)))

    cv2.imshow("Frame", disp_frame)
    cv2.imshow("Mask", disp_mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
