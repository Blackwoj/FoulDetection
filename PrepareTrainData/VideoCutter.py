import cv2
import numpy as np
from pathlib import Path
from .TrainingVideoValidator import TrainingVideoValidator


class VideoSceneSplitter:
    """
    A class for splitting a video into scenes based on histogram differences between frames.
    """
    validator = TrainingVideoValidator()

    def __init__(self, video_path: Path, output_folder: Path, threshold_multiplier: float = 1.5):
        """
        Initialize the VideoSceneSplitter.

        :param video_path: Path to the input video file.
        :param output_folder: Path to the folder where the output scenes will be saved.
        :param threshold_multiplier: Multiplier for the threshold used to detect scene changes.
        """
        self.video_path = video_path
        self.output_folder = output_folder
        self.threshold_multiplier = threshold_multiplier

    def histogram_difference(self, hist1, hist2) -> float:
        """
        Calculate the difference between two histograms.

        :param hist1: First histogram.
        :param hist2: Second histogram.
        :return: Histogram difference value.
        """
        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        return diff

    def calculate_histogram(self, frame) -> np.ndarray:
        """
        Calculate the normalized histogram of a frame.

        :param frame: Input frame.
        :return: Normalized histogram.
        """
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def scene_based_splitting(self):
        """
        Split the video into scenes based on histogram differences.
        """
        video_capture = cv2.VideoCapture(str(self.video_path))

        prev_frame = None
        prev_hist = None
        scene_start = 0

        differences = []

        scene_count = 0
        act_frames = []
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            hist = self.calculate_histogram(frame)

            if prev_frame is not None:
                diff = self.histogram_difference(prev_hist, hist)
                differences.append(diff)

                if diff > self.threshold_multiplier:
                    if scene_start > 0:
                        if self.validator.predict_img(act_frames):
                            act_frames = []
                            self.validator
                            self.save_scene(
                                scene_start,
                                int(video_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1, scene_count
                            )
                            scene_count += 1
                        scene_start = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            act_frames.append(frame)
            prev_frame = frame
            prev_hist = hist

        video_capture.release()
        cv2.destroyAllWindows()

    def save_scene(self, start_frame: int, end_frame: int, scene_count: int):
        """
        Save a scene to a video file.

        :param start_frame: Starting frame index of the scene.
        :param end_frame: Ending frame index of the scene.
        :param scene_count: Scene count (used in the output filename).
        """
        video_capture = cv2.VideoCapture(str(self.video_path))

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        filename = f"{self.video_path.stem}_scene_{scene_count}.mp4"
        output_filepath = self.output_folder / filename

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_filepath), fourcc, fps, (width, height))

        for _ in range(start_frame, end_frame + 1):
            ret, frame = video_capture.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        video_capture.release()


# TODO REWRITE TO USE AS method and connect To data loader.
base_path = Path('VideoCutter.py').resolve().parent.parent
data_path = base_path / 'Dirty & Brutal Fouls in Football.mp4'
output_path = base_path / '!Studia' / 'ProjektOutputData'

video_splitter = VideoSceneSplitter(Path('Dirty & Brutal Fouls in Football.mp4'), output_path)
video_splitter.scene_based_splitting()
