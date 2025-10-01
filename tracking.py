from struct import Struct

import torch
import cv2
import numpy as np
from prettytable import PrettyTable
import sys
import os

from new_cd import new_cd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'alltracker'))

from alltracker.nets.alltracker import Net

from alltracker.demo import count_parameters, run

def run_tracking_model(video_path: str, tiny = True) -> None:
    torch.set_grad_enabled(False)
    window_len = 16
    if tiny:
        model = Net(window_len, use_basicencoder=True, no_split=True)
    else:
        model = Net(window_len)

    count_parameters(model)
    args = Args(window_len)

    with new_cd('alltracker'):
        run(model, args)

class Args:
    def __init__(self, window_len):
            self.ckpt_init = ''
            self.mp4_path = 'example_comfy.mp4'
            self.query_frame = 0
            self.image_size = 1024
            self.max_frames = 400
            self.inference_iters = 4
            self.window_len = window_len
            self.rate = 2
            self.conf_thr = 0.1
            self.bkg_opacity = 0.5
            self.vstack = False
            self.hstack = False
            self.tiny = True