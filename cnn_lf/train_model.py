import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

MAX_SPEED = 88.11


class AugmentedInMemoryDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.images = []
        self.speeds = []

        for _, row in df.iterrows():
            img = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = img.astype(np.float32) / 255.0
                left = row['left_speed'] / MAX_SPEED
                right = row['right_speed'] / MAX_SPEED

                self.images.append(img)
                self.speeds.append([left, right])

                mirrored_img = cv2.flip(img, 1)
                self.images.append(mirrored_img)
                self.speeds.append([right, left])

        self.images = np.array(self.images)
        self.speeds = np.array(self.speeds, dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.expand_dims(self.images[idx], axis=0)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(self.speeds[idx], dtype=torch.float32)


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

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 60, 320)
            self.flatten_size = self.features(dummy).numel()

        self.controller = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.controller(x)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = AugmentedInMemoryDataset('driving_data_expert.csv')
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = LineFollowingCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 15
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {epoch_loss:.6f}")

    torch.save(model.state_dict(), 'cozmo_cnn_model.pth')
    with open('model_config.txt', 'w') as f:
        f.write(f"max_speed:{MAX_SPEED}")

    print("Model saved to cozmo_cnn_model.pth.")
