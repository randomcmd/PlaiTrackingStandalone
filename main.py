import argparse
import os
import sys

from depth import run_depth_model
from tracking import run_tracking_model

def main():
    parser = argparse.ArgumentParser(
        prog='tracking',
        description='tracks a singular point on a video in 3d space',
        epilog='this is meant for a comfyui node to spawn'
    )
    parser.add_argument('video', help='the video file to track')
    parser.add_argument('x', help='the x coordinate of the tracking point')
    parser.add_argument('y', help='the y coordinate of the tracking point')
    args = parser.parse_args()

    print(f'Ran program with args: {args.video=} {args.x=} {args.y=}', file=sys.stderr)

    run_depth_model()
    run_tracking_model(video_path=os.path.abspath(args.video))

    sys.stdout.write("[[0, 0, 0]]")
    sys.stdout.newline()

if __name__ == '__main__':
    main()