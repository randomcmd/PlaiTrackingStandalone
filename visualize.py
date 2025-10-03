import cv2
import numpy as np
import torch


def visualize(video_path: str, data: torch.Tensor, output_path: str):
    source = cv2.VideoCapture(video_path)
    fps = source.get(cv2.CAP_PROP_FPS)
    resolution = [int(source.get(cv2.CAP_PROP_FRAME_WIDTH)), int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    output = cv2.VideoWriter(output_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, resolution)

    initial_depth = data[0, 2].int().item()

    i = 0
    while True:
        ret, frame = source.read()
        if not ret:
            break

        tracking_xy = data[i, 0:2].int().tolist()
        tracking_depth = inv_to_metric(data[i, 2]).item()
        tracking_depth_previous = inv_to_metric(data[i-1, 2]).item() if i > 0 else initial_depth

        radius = 50
        radius_depth_adjusted = scale_radius(
            initial_radius=radius,
            initial_depth=initial_depth,
            depth=tracking_depth,
            min_radius=10,
            smooth_factor=1.0,
            previous_depth=tracking_depth_previous,
        )

        frame = cv2.circle(frame, tracking_xy, 10, color=(255, 0, 0), thickness=2)
        frame = cv2.circle(frame, tracking_xy, int(radius_depth_adjusted), color=(255, 0, 0), thickness=2)

        output.write(frame)
        i += 1

    source.release()
    output.release()

def scale_radius(initial_radius: float,
                 initial_depth:   float,
                 depth:    float,
                 min_radius:   float = 1.0,
                 smooth_factor: float = 0.0,
                 previous_depth:   float = None) -> float:
    # Guard against division by zero (or values that are too close to 0)
    eps = np.finfo(float).eps
    depth = max(depth, eps)

    # Optional temporal smoothing – reduces flicker if the depth map is noisy
    if smooth_factor > 0 and previous_depth is not None:
        depth = smooth_factor * depth + (1 - smooth_factor) * previous_depth

    # Inverse‑depth scaling factor
    factor = initial_depth / depth

    # Apply the factor to the original radius
    new_radius = initial_radius * factor

    # Enforce a minimum size so the circle never disappears
    new_radius = max(min_radius, new_radius)

    # Return an integer pixel count (most drawing APIs expect int)
    return int(round(new_radius))

def inv_to_metric(inv_depth: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Turn inverse‑depth (or disparity) into metric depth."""
    return 1.0 / (inv_depth.clamp(min=eps))