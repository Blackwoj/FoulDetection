# pylint: disable=E1101

import cv2
import logging
import os
from .subProcessing.PlayersMarking import MarkPlayer
from .subProcessing.PoseDetection import PoseDetector
from typing import Any, List, Union


class DataProcessor:

    def __init__(self) -> None:
        self.mark_player = MarkPlayer()
        self.pose_detector = PoseDetector()

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

    def process_frames(self, frames: list[Any]) -> Union[List[Any], None]:
        data_to_train = []
        new_video = 1
        for i, frame in enumerate(frames, start=1):
            act_row = []
            self.pose_detector.rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result, box_of_interest, _, players_relative, players_centers = self.mark_player.predict_img(frame, new_video)
            new_video = 0

            if result is not None and players_centers is not None:
                pose_landmarks_list = self.pose_detector.detect_pose_landmarks(frame, box_of_interest, players_relative)
                p1_center = self.normalize_players_centers(players_centers[0].tolist(), frame.shape)
                p2_center = self.normalize_players_centers(players_centers[1].tolist(), frame.shape)
                if len(pose_landmarks_list[0]) == 1 and len(pose_landmarks_list[1]) == 1:
                    act_row = self.prepare_data_to_train(pose_landmarks_list)
                    act_row[0].append(p1_center[0])
                    act_row[0].append(p1_center[1])
                    act_row[0].append(p2_center[0])
                    act_row[0].append(p2_center[1])
                    data_to_train.append(act_row[0])
        return data_to_train

    def prepare_data_to_train(self, data):
        landmarks_data = []
        for player in data:
            player_data = []
            for landmarks_frame in player:
                landmarks_frame_data = [landmark.x for landmark in landmarks_frame] + [landmark.y for landmark in landmarks_frame] + [landmark.z for landmark in landmarks_frame]
                player_data.extend(landmarks_frame_data)
            landmarks_data.append(player_data)
        return landmarks_data

    def normalize_players_centers(self, player_center, frame_size):
        norm_x = player_center[0][0] / frame_size[0]
        norm_y = player_center[0][1] / frame_size[1]

        return [norm_x, norm_y]
