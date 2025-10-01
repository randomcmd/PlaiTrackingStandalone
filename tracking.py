import torch
import cv2
import numpy as np
from prettytable import PrettyTable
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'alltracker'))

from alltracker.nets.alltracker import Net

from alltracker.demo import count_parameters

def run_tracking_model(video_path: str, tiny = True) -> None:
    if tiny:
        model = Net(args.window_len, use_basicencoder=True, no_split=True)
        url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker_tiny.pth"
    else:
        model = Net(args.window_len)
        url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"

    count_parameters(model)
    run(model, None)
