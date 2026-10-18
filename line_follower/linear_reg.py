import time
import cv2
import numpy as np
import pycozmo

m = 0.0
b = 160.0
left_turn = 0.0
right_turn = 0.0


def on_camera_image(cli, new_image):
    global m, b, left_turn, right_turn

    gray_frame = np.array(new_image.convert('L'))
    roi = gray_frame[180:240, 0:320]
    _, thresh = cv2.threshold(roi, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    y_indices, x_indices = np.where(thresh == 255)

    if len(x_indices) > 10:
        X_data = y_indices.astype(np.float32)
        Y_target = x_indices.astype(np.float32)

        m, b = np.polyfit(X_data, Y_target, 1)

        predicted_center_x = (m * 60.0) + b
        error = predicted_center_x - 160.0

        base_speed = 20.0
        steering_sensitivity = 0.3
        correction = steering_sensitivity * error

        left_turn = base_speed + correction
        right_turn = base_speed - correction

        debug_view = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        cv2.circle(debug_view, (int(predicted_center_x), 30), 5, (0, 255, 0), -1)
        cv2.imshow("Camera View", thresh)
        cv2.imshow("Line Regression", debug_view)
        cv2.waitKey(1)
    else:
        left_turn = 0.0
        right_turn = 0.0


with pycozmo.connect() as cli:
    cli.set_head_angle(pycozmo.robot.MIN_HEAD_ANGLE.radians)
    time.sleep(1)

    cli.enable_camera(True, color=False)
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, on_camera_image)

    try:
        while True:
            cli.drive_wheels(left_turn, right_turn)
            time.sleep(0.05)
    except KeyboardInterrupt:
        cli.drive_wheels(0, 0)
        cv2.destroyAllWindows()