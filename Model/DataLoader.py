import os
import logging
import glob
import cv2
from pathlib import Path
from .DataProcessor.DataProcessor import DataProcessor
import numpy as np

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
        logging.info("Start DataLoader...")
        file_index = 1
        self.valid_frames_value = []
        for filename in glob.glob(os.path.join(self.data_path, f'*{VIDEO_FILE_EXTENSION}')):
            filepath = os.path.join(self.data_path, filename)
            self.frames = []

            cap = cv2.VideoCapture(filepath)  # type: ignore

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                self.frames.append(frame)
            cap.release()

            logging.info(f"Video {filename} has been read, passed to preproces data.")

            # Call the DataPreProcessor method for the frames
            name, ext = os.path.splitext(filename)
            self.save_data_to_csv(self.data_preprocessor.process_frames(
                self.frames,
                name,
                self.output_folder
            ), file_index)
            file_index += 1
        np.savetxt('valid_frame_value1.csv', self.valid_frames_value, delimiter=", ", fmt='% s')

    def save_data_to_csv(self, data, output_num: int):
        output_file = f"{output_num}.csv"
        project_path = Path(__file__).resolve().parent.parent
        output_path = project_path / "trainData" / 'NoFoul' / output_file
        self.valid_frames_value.append(len(data))
        np.savetxt(
            output_path,
            data,
            delimiter=", ",
            fmt='% s'
        )
        print(f'Data saved to {output_file}')
