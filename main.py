import argparse
import os
import sys
import torch

from depth import run_depth_model
from tracking import run_tracking_model, extract_closest_trajectory


def main():
    parser = argparse.ArgumentParser(
        prog='tracking',
        description='tracks a singular point on a video in 3d space',
        epilog='this is meant for a comfyui node to spawn'
    )
    parser.add_argument('video', help='the video file to track')
    parser.add_argument('x', type=int, help='the x coordinate of the tracking point')
    parser.add_argument('y', type=int, help='the y coordinate of the tracking point')
    args = parser.parse_args()

    print(f'Ran program with args: {args.video=} {args.x=} {args.y=}', file=sys.stderr)

    # TODO: Depth model and tracking model could be running in parallel
    tracking = run_tracking_model(video_path=os.path.abspath(args.video))
    target_trajectory = extract_closest_trajectory(tracking, args.x, args.y)

    depths = run_depth_model(video_path=os.path.abspath(args.video))

    print(f'{target_trajectory=}', file=sys.stderr)
    print(f'{depths=}', file=sys.stderr)

    data = apply_depth_data_to_tracking_data(target_trajectory, depths)
    print(f'{data=}', file=sys.stderr)

    sys.stdout.write("[[0, 0, 0]]\n")

def apply_depth_data_to_tracking_data(tracking: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
    """
        target_trajectory : (B, N, 2)   – pixel coordinates (x, y) of N points per frame
        depths           : (B, 1, H, W) – depth map for each frame
        Returns:
            depth_at_points : (B, N)   – depth value for each trajectory point
        """
    B, N, _ = tracking.shape
    _, _, H, W = depths.shape

    # -------------------------------------------------
    # 1) Normalise to [-1, 1] (remember width ↔ x, height ↔ y)
    grid = tracking.clone().float()
    grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1  # x
    grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1  # y

    # -------------------------------------------------
    # 2) Reshape to (B, N, 1, 2) – treat N points as a “column”
    grid = grid.unsqueeze(2)  # now (B, N, 1, 2)

    # -------------------------------------------------
    # 3) Sample depth values
    sampled = torch.nn.functional.grid_sample(
        depths, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )  # → (B, 1, N, 1)

    # -------------------------------------------------
    # 4) Remove the singleton dimensions
    depth_at_points = sampled.squeeze(1).squeeze(-1)  # (B, N)

    return depth_at_points

if __name__ == '__main__':
    main()