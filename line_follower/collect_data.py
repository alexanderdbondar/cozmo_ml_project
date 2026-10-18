import csv
import threading
import time
import cv2
import numpy as np
import pycozmo
from pynput import keyboard

dataset = []
left_speed = 0.0
right_speed = 0.0
base_speed = 10.0
data_lock = threading.Lock()


def key_press(key):
    global left_speed, right_speed
    with data_lock:
        if key == keyboard.Key.up:
            left_speed, right_speed = base_speed, base_speed
        elif key == keyboard.Key.down:
            left_speed, right_speed = -base_speed, -base_speed
        elif key == keyboard.Key.left:
            left_speed, right_speed = base_speed * 0.1, base_speed * 1.8
        elif key == keyboard.Key.right:
            left_speed, right_speed = base_speed * 1.8, base_speed * 0.1


def on_release(key):
    global left_speed, right_speed
    if key in [keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right]:
        with data_lock:
            left_speed = 0.0
            right_speed = 0.0
    if key == keyboard.Key.esc:
        return False


listener = keyboard.Listener(on_press=key_press, on_release=on_release)
listener.start()


def model_calc(cli, img):
    gray_frame = np.array(img.convert('L'))
    roi = gray_frame[180:240, 0:320]
    _, thresh = cv2.threshold(roi, 60, 255, cv2.THRESH_BINARY_INV)

    y_indices, x_indices = np.where(thresh == 255)
    if len(x_indices) > 10:
        X_data = y_indices.astype(np.float32)
        Y_target = x_indices.astype(np.float32)

        m, b = np.polyfit(X_data, Y_target, 1)
        predicted_center_x = (m * 60.0) + b
        error = 160.0 - predicted_center_x

        with data_lock:
            current_l, current_r = left_speed, right_speed

        if current_l != 0.0 or current_r != 0.0:
            dataset.append([predicted_center_x, current_l, current_r, m, b, error])


def motor_control_loop(cli):
    while listener.running:
        with data_lock:
            current_l, current_r = left_speed, right_speed
        cli.drive_wheels(current_l, current_r)
        time.sleep(0.01)


with pycozmo.connect() as cli:
    cli.set_head_angle(pycozmo.robot.MIN_HEAD_ANGLE.radians)
    time.sleep(1)

    cli.enable_camera(True, color=False)
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, model_calc)

    motor_thread = threading.Thread(target=motor_control_loop, args=(cli,))
    motor_thread.start()

    while listener.running:
        time.sleep(0.1)

    motor_thread.join()

with open('driving_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['predicted_center_x', 'left_speed', 'right_speed', 'm', 'b', 'error'])
    writer.writerows(dataset)

print(f"Saved {len(dataset)} dataset entries to driving_data.csv.")