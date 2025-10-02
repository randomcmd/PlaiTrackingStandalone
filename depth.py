from model_context import model_context
from videodepthanything_demo_modified import run

def run_depth_model(video_path: str):
    with model_context('VideoDepthAnything'):
        args = Args(
            video_path=video_path
        )
        return run(args)

class Args:
    def __init__(self, video_path: str):
        self.input_video = video_path
        self.output_dir='./outputs'
        self.input_size=518
        self.max_res=1280
        self.encoder = 'vitl'
        self.max_len=-1
        self.target_fps=-1
        self.metric = False
        self.fp32 = False
        self.grayscale = False
        self.focal_length_x = 470.4
        self.focal_length_y = 470.4