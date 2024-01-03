import os
import logging
import glob
import cv2
from pathlib import Path
from .DataProcessor import DataProcessor

VIDEO_FILE_EXTENSION = '.mp4'


class DataLoader:
    """Class for loading and processing video data."""

    def __init__(
        self,
        data_location: Path,
        framed_videos: str = os.path.dirname(os.path.realpath(__file__))
    ) -> None:
        """
        Initialize the DataLoader.

        Parameters:
        - data_location (str): The path to the data.
        - framed_videos (str): The path to the framed videos.
        """
        if not os.path.exists(data_location):
            raise KeyError(f"Invalid data path: {data_location}")
        self.data_path = data_location
        self.output_folder = os.path.join(framed_videos, "data", "framed_video")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.frames = []

        # Create an instance of DataPreProcessor
        self.data_preprocessor = DataProcessor()

    def _load_data(self) -> None:
        """
        Load video data and process frames.
        """
        for filename in glob.glob(os.path.join(self.data_path, f'*{VIDEO_FILE_EXTENSION}')):
            filepath = os.path.join(self.data_path, filename)

            cap = cv2.VideoCapture(filepath)  # type: ignore

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                self.frames.append(frame)
            cap.release()

            logging.info(f"Video {filename} has been read, and frames have been saved.")

            # Call the DataPreProcessor method for the frames
            name, ext = os.path.splitext(filename)
            self.data_preprocessor.process_frames(
                self.frames,
                name,
                self.output_folder
            )
