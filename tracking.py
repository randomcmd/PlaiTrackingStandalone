import torch
import sys
import os
from typing import Tuple

from model_context import model_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'alltracker'))

from alltracker.nets.alltracker import Net

from alltracker_demo_modified import count_parameters, run


def run_tracking_model(video_path: str, debug_output=False, tiny=True) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.set_grad_enabled(False)
    window_len = 16
    if tiny:
        model = Net(window_len, use_basicencoder=True, no_split=True)
    else:
        model = Net(window_len)

    count_parameters(model)
    args = Args(
        video_path=video_path,
        tiny=tiny,
        debug_output=debug_output,
        window_len=window_len
    )

    with model_context(working_directory='alltracker'):
        xy, confidence = run(model, args)

    return xy, confidence


class Args:
    def __init__(self, video_path: str, debug_output: bool, tiny: bool, window_len: int):
        self.ckpt_init = ''
        self.mp4_path = video_path
        self.query_frame = 0
        self.max_frames = 9999
        self.inference_iters = 4
        self.window_len = window_len
        self.rate = 20
        self.conf_thr = 0.5
        self.bkg_opacity = 0.5
        self.vstack = False
        self.hstack = False
        self.tiny = tiny
        self.debug_output = debug_output


def extract_closest_trajectory(tracking: torch.Tensor, x: int, y: int) -> torch.Tensor:
    if tracking.ndim != 3 or tracking.shape[2] != 2:
        raise ValueError(f'tracking must have shape (T, N, 2), got {tracking.shape}')

    xy0 = tracking[0]
    target = torch.tensor([x, y], dtype=xy0.dtype, device=xy0.device)

    distance_squared = ((xy0 - target) ** 2).sum(dim=1)
    best_tracker_index = torch.argmin(distance_squared).item()

    return tracking[:, best_tracker_index, :]
