import torch
import cv2
import numpy as np
from prettytable import PrettyTable
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'alltracker'))

from alltracker.nets.alltracker import Net

from alltracker.demo import count_parameters, run

def run_tracking_model(video_path: str, tiny = True) -> None:
    if tiny:
        model = Net(args.window_len, use_basicencoder=True, no_split=True)
        url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker_tiny.pth"
    else:
        model = Net(args.window_len)
        url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"

    count_parameters(model)
    run(
        model,
        {
            "ckpt_init": '',
            "mp4_path": './demo_video/monkey.mp4',
            "query_frame": 0,
            "image_size": 1024,
            "max_frames": 400,
            "inference_iters": 4,
            "window_len": 16,
            "rate": 2,
            "conf_thr": 0.1,
            "bkg_opacity": 0.5,
            "vstack": False,
            "hstack": False,
            "tiny": False,
        }
    )
