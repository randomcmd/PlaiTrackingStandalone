import argparse
import json
import os
import sys
import torch

from depth import run_depth_model
from tracking import run_tracking_model, extract_closest_trajectory
from visualize import visualize
from pathlib import Path


def main(args):
    print(f'Ran program with args: {args.video=} {args.x=} {args.y=} {args.output=}', file=sys.stderr)

    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)

    # TODO: Depth model and tracking model could be running in parallel
    tracking, tracking_confidence = run_tracking_model(video_path=os.path.abspath(args.video), debug_output=args.output, tiny=False)
    target_trajectory = extract_closest_trajectory(tracking, args.x, args.y)

    depths_raw = run_depth_model(video_path=os.path.abspath(args.video), debug_output=args.output)
    depths = torch.from_numpy(depths_raw).to(target_trajectory.device)

    # print(f'{target_trajectory=}', file=sys.stderr)
    # print(f'{depths=}', file=sys.stderr)

    data = apply_depth_data_to_tracking_data(target_trajectory, depths)
    # print(f'{data=}', file=sys.stderr)

    if args.output:
        visualize(args.video, data, tracking_confidence, args.output)

    sys.stdout.write(json.dumps(data.data.tolist()))
    sys.stdout.write('\n')


def apply_depth_data_to_tracking_data(tracking: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
    depth_vol = depths.unsqueeze(1)

    F, H, W = depths.shape
    xs = tracking[:, 0].float()  # pixel x   ∈ [0, W‑1]
    ys = tracking[:, 1].float()  # pixel y   ∈ [0, H‑1]

    xs_norm = 2.0 * (xs + 0.5) / W - 1.0
    ys_norm = 2.0 * (ys + 0.5) / H - 1.0

    grid = torch.stack([xs_norm, ys_norm], dim=1).view(F, 1, 1, 2)

    sampled = torch.nn.functional.grid_sample(
        depth_vol,
        grid,
        mode='bilinear',
        padding_mode="zeros",
    )  # shape (F, 1, 1, 1)

    depth_vals = sampled.squeeze()  # (F)

    # ----- 4️⃣  Assemble final (x, y, depth) tensor -----
    result = torch.stack([xs, ys, depth_vals], dim=1)  # (F, 3)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='tracking',
        description='tracks a singular point on a video in 3d space',
        epilog='this is meant for a comfyui node to spawn'
    )
    parser.add_argument('video', help='the video file to track')
    parser.add_argument('x', type=int, help='the x coordinate of the tracking point')
    parser.add_argument('y', type=int, help='the y coordinate of the tracking point')
    parser.add_argument('--output', help='output directory for debug videos')
    args = parser.parse_args()
    main(args)