import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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


def main():
    df = pd.read_csv('driving_data.csv')
    df['left_speed'] = df['left_speed'].rolling(window=5, min_periods=1, center=True).mean()
    df['right_speed'] = df['right_speed'].rolling(window=5, min_periods=1, center=True).mean()

    pred_center_x = df['predicted_center_x'].values.astype(np.float32)
    error = df['error'].values.astype(np.float32)
    heading_angle = np.arctan(df['m'].values.astype(np.float32))

    left_speed = df['left_speed'].values.astype(np.float32)
    right_speed = df['right_speed'].values.astype(np.float32)

    lag = 3
    pred_center_x_lag = df['predicted_center_x'].shift(lag).fillna(0.5).values.astype(np.float32)
    error_lag = df['error'].shift(lag).fillna(0.0).values.astype(np.float32)
    heading_angle_lag = np.arctan(df['m'].shift(lag).fillna(0.0).values.astype(np.float32))

    X_original = np.stack([pred_center_x, heading_angle, error, pred_center_x_lag, heading_angle_lag, error_lag], axis=1)[lag:]
    Y_original = np.stack([left_speed, right_speed], axis=1)[lag:]

    X_mirrored = X_original.copy()
    Y_mirrored = Y_original.copy()

    X_mirrored[:, 0] = 1.0 - X_original[:, 0]
    X_mirrored[:, 1] = -X_original[:, 1]
    X_mirrored[:, 2] = -X_original[:, 2]
    X_mirrored[:, 3] = 1.0 - X_original[:, 3]
    X_mirrored[:, 4] = -X_original[:, 4]
    X_mirrored[:, 5] = -X_original[:, 5]
    Y_mirrored[:, 0] = Y_original[:, 1]
    Y_mirrored[:, 1] = Y_original[:, 0]

    X_raw = np.vstack((X_original, X_mirrored))
    Y_raw = np.vstack((Y_original, Y_mirrored))

    X_mean, X_std = np.mean(X_raw, axis=0), np.std(X_raw, axis=0) + 1e-8
    Y_mean, Y_std = np.mean(Y_raw, axis=0), np.std(Y_raw, axis=0) + 1e-8

    X_scaled = (X_raw - X_mean) / X_std
    Y_scaled = (Y_raw - Y_mean) / Y_std

    dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(Y_scaled, dtype=torch.float32))
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = AdvancedDrivingNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 1000
    print(f"Training AdvancedDrivingNet on {len(dataset)} samples for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_Y in train_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        if (epoch + 1) % 200 == 0:
            avg_loss = epoch_loss / len(dataset)
            print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), 'driving_model.pth')
    np.savez('scaling_stats.npz', X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std)
    print("Training complete. Model saved to driving_model.pth.")


if __name__ == '__main__':
    main()
