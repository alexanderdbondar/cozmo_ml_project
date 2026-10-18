import os
import time
import cv2
import dlib
import numpy as np
import pycozmo
from safetensors.numpy import load_file

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

left_turn = 0.0
right_turn = 0.0
decision_made = False
chosen_lane_bias = 0.0
stop_robot = False

tensors = load_file(os.path.join(MODULE_DIR, "model.safetensors"))
W_in, b_in = tensors["input_layer.weight"], tensors["input_layer.bias"]
W_mid, b_mid = tensors["mid_layer.weight"], tensors["mid_layer.bias"]
W_out, b_out = tensors["output_layer.weight"], tensors["output_layer.bias"]

svm_path = os.path.join(MODULE_DIR, "detector.svm")
detector = dlib.simple_object_detector(svm_path) if os.path.exists(svm_path) else None


def predict_digit(raw_digit_roi):
    resized = cv2.resize(raw_digit_roi, (20, 20), interpolation=cv2.INTER_AREA)
    padded = cv2.copyMakeBorder(resized, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    X_normalized = padded.astype(np.float32) / 255.0
    X_standardized = (X_normalized - 0.1307) / 0.3081
    X = X_standardized.reshape(1, 784)

    layer1 = np.maximum(0, np.dot(X, W_in.T) + b_in)
    layer2 = np.maximum(0, np.dot(layer1, W_mid.T) + b_mid)
    final_output = np.dot(layer2, W_out.T) + b_out

    return int(np.argmax(final_output, axis=1).item())


def on_camera_image(cli, new_image):
    global left_turn, right_turn, decision_made, chosen_lane_bias, stop_robot

    gray_frame = np.array(new_image.convert('L'))
    _, thresh = cv2.threshold(gray_frame, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    camera_view = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    left_lane_center, right_lane_center, start_x = None, None, None
    scan_row = 210

    for i in range(319):
        current_pixel = thresh[scan_row, i]
        next_pixel = thresh[scan_row, i + 1]
        if current_pixel == 0 and next_pixel == 255:
            start_x = i + 1
        elif current_pixel == 255 and next_pixel == 0 and start_x is not None:
            lane_center = (start_x + i) / 2.0
            if lane_center < 160:
                left_lane_center = lane_center
            else:
                right_lane_center = lane_center

    if decision_made and left_lane_center is not None and right_lane_center is not None:
        lane_width = right_lane_center - left_lane_center
        if 80 < lane_width < 160:
            decision_made = False
            chosen_lane_bias = 0.0

    if not hasattr(on_camera_image, "frame_counter"):
        on_camera_image.frame_counter = 0

    if not decision_made:
        on_camera_image.frame_counter += 1
        dets = []
        if detector and on_camera_image.frame_counter % 10 == 0:
            rgb_frame = np.array(new_image.convert('RGB'))
            dets = detector(rgb_frame)

        for d in dets:
            cv2.rectangle(camera_view, (d.left(), d.top()), (d.right(), d.bottom()), (0, 165, 255), 2)
            stop_robot = True
            decision_made = True
            break

        if not stop_robot and left_lane_center is not None and right_lane_center is not None:
            roi_y1, roi_y2, roi_x1, roi_x2 = 20, 150, 60, 260
            digit_zone = thresh[roi_y1:roi_y2, roi_x1:roi_x2]
            cv2.rectangle(camera_view, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)

            contours, _ = cv2.findContours(digit_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for ctr in contours:
                x, y, w, h = cv2.boundingRect(ctr)
                area = cv2.contourArea(ctr)
                if (100 < area < 800) and (h < 85) and (10 <= w <= 50) and (20 < h <= 75):
                    extracted_digit = digit_zone[y:y + h, x:x + w]
                    predicted_digit = predict_digit(extracted_digit)
                    if predicted_digit == 1:
                        chosen_lane_bias = -50.0
                        decision_made = True
                    elif predicted_digit == 2:
                        chosen_lane_bias = 50.0
                        decision_made = True

    base_speed = 18.0
    steering_sensitivity = 0.45
    target_center_x = 160.0 + chosen_lane_bias

    if left_lane_center is not None and right_lane_center is not None:
        error = ((left_lane_center + right_lane_center) / 2.0) - target_center_x
    elif left_lane_center is not None:
        error = left_lane_center - 160.0 if chosen_lane_bias < 0 else (left_lane_center + 60.0) - target_center_x
    elif right_lane_center is not None:
        error = right_lane_center - 160.0 if chosen_lane_bias > 0 else (right_lane_center - 60.0) - target_center_x
    else:
        error = 0.0

    correction = steering_sensitivity * error
    left_turn = base_speed + correction
    right_turn = base_speed - correction

    if left_lane_center is not None:
        cv2.circle(camera_view, (int(left_lane_center), scan_row), 5, (255, 0, 0), -1)
    if right_lane_center is not None:
        cv2.circle(camera_view, (int(right_lane_center), scan_row), 5, (0, 0, 255), -1)

    cv2.imshow("Binary View", thresh)
    cv2.imshow("Detection View", camera_view)
    cv2.waitKey(1)


with pycozmo.connect() as cli:
    cli.wait_for_robot()

    for _ in range(25):
        cli.move_lift(1.0)
        time.sleep(0.05)

    cli.move_lift(0.0)
    time.sleep(0.5)
    cli.set_head_angle(-0.3)
    time.sleep(0.5)

    cli.enable_camera(True, color=False)
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, on_camera_image)

    try:
        while True:
            if stop_robot:
                l_speed, r_speed = 0, 0
            else:
                l_speed = max(min(left_turn, 45), -45)
                r_speed = max(min(right_turn, 45), -45)
            cli.drive_wheels(l_speed, r_speed)
            time.sleep(0.05)
    except KeyboardInterrupt:
        cli.drive_wheels(0, 0)
        cv2.destroyAllWindows()