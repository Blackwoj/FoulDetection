import os
import logging
import cv2
from .DataProcessor.DataProcessor import DataProcessor

VIDEO_FILE_EXTENSION = '.mp4'


class DataLoader:
    """Class for loading and processing video data."""

    def __init__(self) -> None:
        """
        Initialize the DataLoader.
        """

        self.data_preprocessor = DataProcessor()

    def load_data(self, video_location: str) -> None:
        """
        Load video data from the specified location and process frames.

        Parameters:
        - video_location (str): Path to the video file.
        """
        logging.info("Start DataLoader...")
        if not video_location.lower().endswith(VIDEO_FILE_EXTENSION):
            raise ValueError("Invalid file format. Only .mp4 files are supported.")

        self.frames = []

        cap = cv2.VideoCapture(video_location)  # type: ignore

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            self.frames.append(frame)
        cap.release()

        logging.info(f"Video {video_location} has been read, passed to pre-process data.")

        # Call the DataPreProcessor method for the frames
        name, ext = os.path.splitext(os.path.basename(video_location))
        if self.frames != []:
            return self.data_preprocessor.process_frames(
                self.frames
            )