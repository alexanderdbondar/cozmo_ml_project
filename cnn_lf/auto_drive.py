import time
import cv2
import numpy as np
import pycozmo
import torch
import torch.nn as nn

MAX_SPEED = 88.11
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class LineFollowingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
        )
        self.controller = nn.Sequential(
            nn.Linear(9472, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.controller(x)


model = LineFollowingCNN().to(device)
model.load_state_dict(torch.load('cozmo_cnn_model.pth', map_location=device))
model.eval()
print(f"Loaded CNN model on device: {device}")


def autonomous_driving_loop(cli, img):
    gray_frame = np.array(img.convert('L'))
    roi = gray_frame[180:240, 0:320].astype(np.float32) / 255.0

    img_tensor = torch.tensor(np.expand_dims(roi, axis=(0, 1)), dtype=torch.float32).to(device)

    with torch.no_grad():
        normalized_predictions = model(img_tensor)
        speeds = normalized_predictions.cpu().numpy()[0]

    left_speed = np.clip(speeds[0] * MAX_SPEED, -MAX_SPEED, MAX_SPEED)
    right_speed = np.clip(speeds[1] * MAX_SPEED, -MAX_SPEED, MAX_SPEED)

    cli.drive_wheels(float(left_speed), float(right_speed))


with pycozmo.connect() as cli:
    cli.set_head_angle(pycozmo.robot.MIN_HEAD_ANGLE.radians)
    time.sleep(1)

    print("Autonomous CNN mode active. Press Ctrl+C to stop.")
    cli.enable_camera(True, color=False)
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, autonomous_driving_loop)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping Cozmo...")
        cli.drive_wheels(0.0, 0.0)
