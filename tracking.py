import torch
import cv2
import argparse
import utils.saveload
import utils.basic
import utils.improc
import PIL.Image
import numpy as np
import os
from prettytable import PrettyTable
import time
from nets.alltracker import Net

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
