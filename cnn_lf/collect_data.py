import csv
import os
import time
import cv2
import numpy as np
import pycozmo

os.makedirs('images_expert', exist_ok=True)

dataset = []
frame_count = 0
m = 0.0
b = 160.0
left_turn = 0.0
right_turn = 0.0


def on_camera_image(cli, new_image):
    global m, b, left_turn, right_turn, frame_count

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

        if left_turn != 0.0 or right_turn != 0.0:
            filename = f"images_expert/frame_{frame_count:06d}.png"
            cv2.imwrite(filename, roi)
            dataset.append([filename, left_turn, right_turn])
            frame_count += 1

        debug_view = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        cv2.circle(debug_view, (int(predicted_center_x), 30), 5, (0, 255, 0), -1)
        cv2.imshow("Camera View", thresh)
        cv2.imshow("Line Regression", debug_view)
        cv2.waitKey(1)
    else:
        left_turn = 0.0
        right_turn = 0.0


try:
    with pycozmo.connect() as cli:
        cli.set_head_angle(pycozmo.robot.MIN_HEAD_ANGLE.radians)
        time.sleep(1)

        cli.enable_camera(True, color=False)
        cli.add_handler(pycozmo.event.EvtNewRawCameraImage, on_camera_image)

        print("Recording data. Press Ctrl+C to finish.")
        while True:
            cli.drive_wheels(left_turn, right_turn)
            time.sleep(0.05)
except KeyboardInterrupt:
    print("\nStopping robot and saving dataset...")

if dataset:
    with open('driving_data_expert.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_path', 'left_speed', 'right_speed'])
        writer.writerows(dataset)
    print(f"Saved {frame_count} frames to driving_data_expert.csv.")
else:
    print("No frames were recorded.")
