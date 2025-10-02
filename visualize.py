import cv2
import torch


def visualize(video_path: str, data: torch.Tensor, output_path: str):
    source = cv2.VideoCapture(video_path)
    fps = source.get(cv2.CAP_PROP_FPS)
    resolution = [int(source.get(cv2.CAP_PROP_FRAME_WIDTH)), int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    output = cv2.VideoWriter(output_path, -1, fps, resolution)

    while True:
        ret, frame = source.read()
        if not ret:
            break

        frame = cv2.circle(frame, [100, 100], 10, color=(255, 0, 0), thickness=2)

        output.write(frame)

    source.release()
    output.release()