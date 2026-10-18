import sys
import time
import cv2
import numpy as np
import pycozmo
import torch
import torch.nn as nn


class AdvancedDrivingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(6, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.hidden3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 2)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leaky_relu(self.hidden1(x))
        x = self.leaky_relu(self.hidden2(x))
        x = self.leaky_relu(self.hidden3(x))
        return self.output(x)


model = AdvancedDrivingNet()
try:
    model.load_state_dict(torch.load('driving_model.pth'))
    model.eval()
    stats = np.load('scaling_stats.npz')
    X_mean, X_std = stats['X_mean'], stats['X_std']
    Y_mean, Y_std = stats['Y_mean'], stats['Y_std']
    print("Loaded model and scaling stats successfully.")
except FileNotFoundError:
    print("Error: Required model or stats file not found.")
    sys.exit(1)

m = 0.0
b = 160.0
left_turn = 0.0
right_turn = 0.0
history_queue = []


def on_camera_image(cli, new_image):
    global m, b, left_turn, right_turn, history_queue

    gray_frame = np.array(new_image.convert('L'))
    roi = gray_frame[180:240, 0:320]

    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    y_indices, x_indices = np.where(thresh == 255)

    if len(x_indices) > 10:
        X_data = y_indices.astype(np.float32)
        Y_target = x_indices.astype(np.float32)

        m, b = np.polyfit(X_data, Y_target, 1)

        predicted_center_x = (m * 60.0) + b
        error = predicted_center_x - 160.0
        heading_angle = np.arctan(m)
        scaled_center_x = predicted_center_x / 320.0

        history_queue.append([scaled_center_x, heading_angle, error])
        while len(history_queue) > 4:
            history_queue.pop(0)
        while len(history_queue) < 4:
            history_queue.append([scaled_center_x, heading_angle, error])

        lagged = history_queue[0]
        raw_inputs = np.array([
            scaled_center_x, heading_angle, error,
            lagged[0], lagged[1], lagged[2]
        ], dtype=np.float32)

        scaled_inputs = (raw_inputs - X_mean) / X_std
        input_tensor = torch.tensor(scaled_inputs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            scaled_predictions = model(input_tensor).squeeze(0).cpu().numpy()

        predictions = (scaled_predictions * Y_std) + Y_mean
        left_turn = float(predictions[0]) / 2.0
        right_turn = float(predictions[1]) / 2.0

        debug_view = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        cv2.circle(debug_view, (int(predicted_center_x), 30), 5, (0, 255, 0), -1)
        cv2.imshow("Camera View", thresh)
        cv2.imshow("Line Regression", debug_view)
        cv2.waitKey(1)
    else:
        left_turn = 0.0
        right_turn = 0.0
        cv2.imshow("Camera View", thresh)
        cv2.waitKey(1)


with pycozmo.connect() as cli:
    cli.set_head_angle(pycozmo.robot.MIN_HEAD_ANGLE.radians)
    cli.set_lift_height(92)
    time.sleep(1)

    cli.enable_camera(True, color=False)
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, on_camera_image)

    print("Control loop started. Press Ctrl+C to stop.")
    try:
        while True:
            cli.drive_wheels(left_turn, right_turn)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping Cozmo...")
        cli.drive_wheels(0, 0)
        cv2.destroyAllWindows()
