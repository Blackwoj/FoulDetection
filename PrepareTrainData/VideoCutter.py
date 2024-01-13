import cv2
import numpy as np
from pathlib import Path
from TrainingVideoValidator import TrainingVideoValidator


class VideoSceneSplitter:
    """
    A class for splitting a video into scenes based on histogram differences between frames.
    """
    validator = TrainingVideoValidator()

    def __init__(self, video_path: Path, output_folder: Path, threshold_multiplier: float = 20):
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
        video_capture = cv2.VideoCapture(str(self.video_path))

        prev_frame = None
        prev_hist = None
        third_hist = None
        seconde_hist = None

        differences = []

        scene_count = 0
        act_frames = []
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            hist = self.calculate_histogram(frame)
            if seconde_hist is None:
                seconde_hist = hist
            if third_hist is None:
                third_hist = hist

            if prev_frame is not None:
                diff = np.mean(np.array([self.histogram_difference(prev_hist, hist), self.histogram_difference(seconde_hist, hist), self.histogram_difference(third_hist, hist)]))
                # diff = np.mean(np.array([self.histogram_difference(prev_hist, hist), self.histogram_difference(seconde_hist, hist)]))
                # diff = self.histogram_difference(prev_hist, hist)
                differences.append(diff)
                if int(diff) > self.threshold_multiplier:
                    if self.validator.predict_img(act_frames):
                        print("valid video")
                        self.save_scene(
                            act_frames,
                            scene_count
                        )
                        act_frames = []
                        scene_count += 1
                        seconde_hist = hist
                        third_hist = hist
            act_frames.append(frame)
            prev_frame = frame

            third_hist = seconde_hist
            seconde_hist = prev_hist
            prev_hist = hist

        video_capture.release()
        cv2.destroyAllWindows()

    def save_scene(self, frames: list, scene_count: int):
        """
        Save a scene to a video file.

        :param frames: List of frames for the scene.
        :param scene_count: Scene count (used in the output filename).
        """
        if not frames:
            return
        print(f"save {scene_count}")
        video_capture = cv2.VideoCapture(str(self.video_path))
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        width = int(frames[0].shape[1])  # Assuming the frames have the same width and height
        height = int(frames[0].shape[0])

        filename = f"{self.video_path.stem}_scene_{scene_count}.mp4"
        output_filepath = self.output_folder / filename

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_filepath), fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()
        video_capture.release()


base_path = Path(__file__).resolve().parent.parent.parent.parent
data_path = Path(__file__).resolve().parent.parent / 'videos_to_cut'
output_path = base_path / '!Studia' / 'ProjektOutputData'

for file_path in data_path.glob('*'):  # Iterating through all files in data_path
    if file_path.is_file():  # Checking if the path is a file (not a directory)
        video_splitter = VideoSceneSplitter(file_path, output_path)
        video_splitter.scene_based_splitting()
