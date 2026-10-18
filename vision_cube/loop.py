import time
import cv2
import numpy as np
import pycozmo

SLOPE_M = 13564.6634
INTERCEPT_B = -12.5160

AUTONOMOUS_MODE = True
TARGET_DISTANCE_CM = 10.0
KP_GAIN = 15.0

WIDTH_HISTORY = []
BUFFER_SIZE = 5


def collection_stream(cli, image):
    global WIDTH_HISTORY

    frame = np.array(image)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        WIDTH_HISTORY.append(w)
        if len(WIDTH_HISTORY) > BUFFER_SIZE:
            WIDTH_HISTORY.pop(0)

        smoothed_w = sum(WIDTH_HISTORY) / len(WIDTH_HISTORY)
        linear_feature = 1.0 / smoothed_w
        predicted_distance_cm = ((SLOPE_M * linear_feature) + INTERCEPT_B) / 10.0

        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{predicted_distance_cm:.1f} cm"
        cv2.putText(bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if AUTONOMOUS_MODE:
            error_cm = predicted_distance_cm - TARGET_DISTANCE_CM
            if error_cm > 0.5:
                motor_speed = min(int(error_cm * KP_GAIN), 120)
                cli.drive_wheels(motor_speed, motor_speed)
            elif error_cm < -0.5:
                cli.drive_wheels(-30, -30)
            else:
                cli.drive_wheels(0, 0)
    else:
        if AUTONOMOUS_MODE:
            cli.drive_wheels(0, 0)

    cv2.imshow("Closed-Loop Tracker", bgr)
    cv2.waitKey(1)


with pycozmo.connect() as cli:
    cli.set_head_angle(0.2)
    cli.set_head_light(True)
    cli.set_all_backpack_lights(pycozmo.lights.white_light)
    time.sleep(1.0)

    cli.enable_camera(enable=True, color=True)
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, collection_stream)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        cli.set_head_light(False)
        cli.drive_wheels(0, 0)
        cv2.destroyAllWindows()
