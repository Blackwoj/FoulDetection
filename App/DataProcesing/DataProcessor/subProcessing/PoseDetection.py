import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseDetector:
    def __init__(self):
        # Inicjalizacja detektora
        base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.rgb_frame = []

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        return annotated_image

    def detect_pose_landmarks(self, frame, box_of_interest, players_relative):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = []
        for player_area in players_relative:

            # Convert absolute coordinates to frame coordinates
            x1_frame, y1_frame, x2_frame, y2_frame = self.decode_location(player_area, box_of_interest)
            player_frame = rgb_frame[y1_frame:y2_frame, x1_frame:x2_frame]

            # Tworzenie obiektu mp.Image z zakodowanego obrazu
            player_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(player_frame))
            player_detection_result = self.detector.detect(player_image)
            landmarks.append(player_detection_result.pose_landmarks)
        return landmarks

    def overlay_landmarks_on_frame(self, frame, player_areas, box_of_interest, landmarks_for_players):
        overlay_frame = frame.copy()

        for player_area, landmarks in zip(player_areas, landmarks_for_players):
            x1_frame, y1_frame, x2_frame, y2_frame = self.decode_location(player_area, box_of_interest)

            # Check if landmarks are available for the player
            if landmarks:
                player_overlay_frame = self.draw_landmarks_on_image(frame[y1_frame:y2_frame, x1_frame:x2_frame], landmarks)
                # Nakładanie fragmentu z landamrkami na oryginalny obraz
                overlay_frame[int(y1_frame):int(y2_frame), int(x1_frame):int(x2_frame)] = player_overlay_frame

        return overlay_frame

    def close(self):
        self.detector.close()

    def decode_location(self, relative, base):
        x1, y1, x2, y2 = relative.numpy()

        # Convert absolute coordinates to frame coordinates
        frame_width, frame_height, _, _ = base
        x1_frame = int(x1 + frame_width)
        y1_frame = int(y1 + frame_height)
        x2_frame = int(x2 + frame_width)
        y2_frame = int(y2 + frame_height)
        return x1_frame, y1_frame, x2_frame, y2_frame