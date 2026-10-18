import numpy as np


def main():
    distances = np.array([50.0, 100.0, 150.0, 200.0])
    pixel_widths = np.array([212.0, 120.0, 86.0, 63.0])

    X_transformed = 1.0 / pixel_widths
    Y_targets = distances

    slope_m, intercept_b = np.polyfit(X_transformed, Y_targets, 1)

    print(f"Fitted Slope (m): {slope_m:.4f}")
    print(f"Fitted Intercept (b): {intercept_b:.4f}")


if __name__ == '__main__':
    main()