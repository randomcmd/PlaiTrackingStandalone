import cv2
import torch


def visualize(video_path: str, data: torch.Tensor, output_path: str):
    source = cv2.VideoCapture(video_path)
    fps = source.get(cv2.CAP_PROP_FPS)
    resolution = [int(source.get(cv2.CAP_PROP_FRAME_WIDTH)), int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    output = cv2.VideoWriter(output_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, resolution)

    initial_depth = data[0, 2].int().item()

    i = 0
    while True:
        ret, frame = source.read()
        if not ret:
            break

        tracking_xy = data[i, 0:2].int().tolist()
        tracking_depth = data[i, 2].int().item()

        frame = cv2.circle(frame, tracking_xy, 10, color=(255, 0, 0), thickness=2)
        frame = cv2.circle(frame, tracking_xy, int(tracking_depth/initial_depth), color=(255, 0, 0), thickness=2)

        output.write(frame)
        i += 1

    source.release()
    output.release()