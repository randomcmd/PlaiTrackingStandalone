# Tracking Standalone for Plai

This scripts aim to simplify the tracking process, by having one script that you can execute from ComfyUI or an API.

## Tracking Steps

1. Run VideoDepthAnything-Small on the video to get video depth
2. Run AllTracker to track the relevant point
3. Combine Depth information with tracking information

## Usage
Run `./setup.sh` on the runpod to set up dependencies and download model weights. `main.py` is the tracking script. Run `python main.py` without any args or with -h to get help on the different arguments.