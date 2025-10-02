# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import numpy as np
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'VideoDepthAnything'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'VideoDepthAnything', 'video_depth_anything'))

from VideoDepthAnything.video_depth_anything.video_depth import VideoDepthAnything
from VideoDepthAnything.utils.dc_utils import read_video_frames, save_video

def run():
    args = Args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder], metric=args.metric)
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/{checkpoint_name}_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

    video_name = os.path.basename(args.input_video)
    os.makedirs(args.output_dir, exist_ok=True)

    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
    depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')
    save_video(frames, processed_video_path, fps=fps)
    save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)

    if args.save_npz:
        depth_npz_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths.npz')
        np.savez_compressed(depth_npz_path, depths=depths)
    if args.save_exr:
        depth_exr_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_exr')
        os.makedirs(depth_exr_dir, exist_ok=True)
        import OpenEXR
        import Imath
        for i, depth in enumerate(depths):
            output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
            header = OpenEXR.Header(depth.shape[1], depth.shape[0])
            header["channels"] = {
                "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            exr_file = OpenEXR.OutputFile(output_exr, header)
            exr_file.writePixels({"Z": depth.tobytes()})
            exr_file.close()

    if args.metric:
        import open3d as o3d

        width, height = depths[0].shape[-1], depths[0].shape[-2]
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - width / 2) / args.focal_length_x
        y = (y - height / 2) / args.focal_length_y

        for i, (color_image, depth) in enumerate(zip(frames, depths)):
            z = np.array(depth)
            points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
            colors = np.array(color_image).reshape(-1, 3) / 255.0

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(os.path.join(args.output_dir, 'point' + str(i).zfill(4) + '.ply'), pcd)

class Args:
    def __init__(self):
        self.input_video = './assets/example_videos/davis_rollercoaster.mp4'
        self.output_dir='./outputs'
        self.input_size=518
        self.max_res=1280
        self.encoder = 'vitl'
        self.max_len=-1
        self.target_fps=-1,
        self.metric = False
        self.fp32 = False
        self.grayscale = False
        self.save_npz = False
        self.save_exr = False
        self.focal_length_x = 470.4
        self.focal_length_y = 470.4