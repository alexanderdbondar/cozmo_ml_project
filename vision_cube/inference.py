import time
import cv2
import numpy as np
import pycozmo

SLOPE_M = 13564.6634
INTERCEPT_B = -12.5160


def collection_stream(cli, image):
    frame = np.array(image)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -5)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        linear_feature = 1.0 / w
        predicted_distance_mm = (SLOPE_M * linear_feature) + INTERCEPT_B
        predicted_distance_cm = predicted_distance_mm / 10.0

        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{predicted_distance_cm:.1f} cm"
        cv2.putText(bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Distance Tracker", bgr)
    cv2.waitKey(1)


with pycozmo.connect() as cli:
    cli.set_head_angle(0.1)
    cli.enable_camera(enable=True, color=True)
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, collection_stream)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()