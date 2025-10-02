import argparse
import os
import sys

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

    sys.stdout.write("[[0, 0, 0]]\n")

if __name__ == '__main__':
    main()