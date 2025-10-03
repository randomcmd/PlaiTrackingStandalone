import argparse
import json
import os
import sys
import torch

from depth import run_depth_model
from tracking import run_tracking_model, extract_closest_trajectory
from visualize import visualize


def main():
    parser = argparse.ArgumentParser(
        prog='tracking',
        description='tracks a singular point on a video in 3d space',
        epilog='this is meant for a comfyui node to spawn'
    )
    parser.add_argument('video', help='the video file to track')
    parser.add_argument('x', type=int, help='the x coordinate of the tracking point')
    parser.add_argument('y', type=int, help='the y coordinate of the tracking point')
    parser.add_argument('--output', help='outputs debug video')
    args = parser.parse_args()

    print(f'Ran program with args: {args.video=} {args.x=} {args.y=}', file=sys.stderr)

    # TODO: Depth model and tracking model could be running in parallel
    tracking = run_tracking_model(video_path=os.path.abspath(args.video), debug_output=args.output is not None, tiny=False)
    target_trajectory = extract_closest_trajectory(tracking, args.x, args.y)

    depths_raw = run_depth_model(video_path=os.path.abspath(args.video), debug_output=args.output is not None)
    depths = torch.from_numpy(depths_raw).to(target_trajectory.device)

    print(f'{target_trajectory=}', file=sys.stderr)
    print(f'{depths=}', file=sys.stderr)

    data = apply_depth_data_to_tracking_data(target_trajectory, depths)
    print(f'{data=}', file=sys.stderr)

    if args.output:
        visualize(args.video, data, args.output)

    sys.stdout.write(json.dumps(data.data.tolist()))
    sys.stdout.write('\n')

def apply_depth_data_to_tracking_data(tracking: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
    B, N, = tracking.shape
    F, H, W = depths.shape
    # TODO: Points should be able to go out of frame
    depth_at_points = torch.stack([
        tracking[:, 0].long().clamp(0, W - 1).float(),
        tracking[:, 1].long().clamp(0, H - 1).float(),
        depths[
            torch.arange(F, device=tracking.device),  # frame indices
            tracking[:, 1].long().clamp(0, H - 1),  # y
            tracking[:, 0].long().clamp(0, W - 1)  # x
        ]
    ], dim=1)  # → (F, 3) on GPU
    return depth_at_points

if __name__ == '__main__':
    main()