# pylint: disable=E1101

import cv2
import logging
import os
from .subProcessing.PlayersMarking import MarkPlayer
from .subProcessing.PoseDetection import PoseDetector
from .Visualizer import Visualizer


class DataProcessor:

    def __init__(self) -> None:
        self.mark_player = MarkPlayer()
        self.pose_detector = PoseDetector()  # Dodano inicjalizację klasy PoseDetector
        self.visualizer = Visualizer()

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

        marked_frames = []
        landmark_frame = []
        i = 0
        for frame in frames:
            i += 1
            self.pose_detector.rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result, box_of_interest, ball_relative, players_relative = self.mark_player.predict_img(frame)
            # marked_frames.append(self.visualizer.draw_bounding_box(frame, box_of_interest, "FoulArea"))
            pose_landmarks_list = self.pose_detector.detect_pose_landmarks(frame, box_of_interest, players_relative)
            landmark_frame = (self.pose_detector.overlay_landmarks_on_frame(frame, players_relative, box_of_interest, pose_landmarks_list))
            marked_frames.append(self.visualizer.draw_bounding_box(landmark_frame, box_of_interest, "FoulArea"))
        self.create_output_video(marked_frames, output_folder, f"{filename}_done_video")
        # self.create_output_video(landmark_frame, output_folder, f"{filename}_landmark")
