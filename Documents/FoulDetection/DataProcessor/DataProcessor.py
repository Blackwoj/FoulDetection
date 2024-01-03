import cv2
import logging
import os
from .subProcessing.PlayersMarking import MarkPlayer  # Import the MarkPlayer class


class DataProcessor:

    def __init__(self) -> None:
        self.frame_drop = FrameDrop()
        self.mark_player = MarkPlayer()  # Initialize MarkPlayer

    def create_output_video(
        self,
        processed_frames,
        output_folder,
        filename: str = "output_video.mp4"
    ):
        logging.info("Creating preprocessed video")

        first_frame = processed_frames[0]
        num_channels = first_frame.shape[-1] if len(first_frame.shape) > 2 else 1

        height, width = first_frame.shape[:2]
        fps = 30

        filename = f"{filename}_processed_video.mp4"
        output_filepath = os.path.join(output_folder, os.path.basename(filename))

        if num_channels == 1:
            video = cv2.VideoWriter(
                output_filepath,
                cv2.VideoWriter_fourcc(*'mp4v'),  # type: ignore
                fps,
                (width, height),
                isColor=False
            )
        else:
            video = cv2.VideoWriter(
                output_filepath,
                cv2.VideoWriter_fourcc(*'mp4v'),  # type: ignore
                fps,
                (width, height),
                isColor=True
            )

        for frame in processed_frames:
            video.write(frame)

        video.release()

    def process_frames(self, frames: list, filename, output_folder):
        processed_frames = self.frame_drop.similar_frame_drop(frames)

        # Pass frames through MarkPlayer for predictions
        marked_frames = []
        for frame in processed_frames:
            predictions = self.mark_player.predict_img(frame)
            marked_frames.append(predictions)

        self.create_output_video(marked_frames, output_folder, f"{filename}_sobel_processed")