import yolov5
import numpy as np
import torch
import cv2
import colorsys
import logging
from typing import List


class MarkPlayer:

    def __init__(self):
        self.model = yolov5.load('keremberke/yolov5m-football')
        self.model.conf = 0.25  # type: ignore NMS confidence threshold
        self.model.iou = 0.45  # type: ignore NMS IoU threshold
        self.model.agnostic = False  # type: ignore NMS class-agnostic
        self.model.multi_label = False  # type: ignore NMS multiple labels per box
        self.model.max_det = 1000  # type: ignore maximum number of detections per image
        self.colors = []
        self.previous_area = None

    def predict_img(self, img):
        results = self.model(img, size=640)
        previous_area = self.define_valuable_area(results.pred[0], img)
        return results, previous_area

    def define_valuable_area(self, prediction, img):
        found_players = {}
        player = 0
        ball = 0

        for element in prediction:
            if element[5] == 1:
                player += 1
            else:
                ball += 1
        if player < 2 or ball != 1:
            return self.widen_area() if self.previous_area else None

        ball_category = prediction[prediction[:, -1] == 0]
        ball_centers = self.center_of_area(ball_category[0])

        players_category = prediction[prediction[:, -1] == 1]

        colored_players = []
        for player in players_category:
            player_color = self._skirt_color(img, player)
            self.colors.append(player_color[0])
            colored_players.append([player_color[0], player])

        self.colors.sort()
        if len(self.colors) >= 2:
            teams_colors = [self.colors[0], self.colors[-1]]
        else:
            logging.error("Cannot define 2 players")
            return None

        for player in colored_players:
            player_area_center = self.center_of_area(player[1])
            distances_to_ball = torch.cdist(player_area_center.unsqueeze(0), ball_centers)
            player_team_diff = [player[0] - team for team in teams_colors]
            player_team_index = min(range(len(player_team_diff)), key=lambda i: abs(player_team_diff[i]))

            if len(found_players) >= 1:
                if player_team_index not in found_players:
                    found_players[player_team_index] = [distances_to_ball, player[1]]
                elif found_players[player_team_index][0] > distances_to_ball:
                    found_players[player_team_index] = [distances_to_ball, player[1]]
            else:
                found_players[player_team_index] = [distances_to_ball, player[1]]

        return self.define_area(ball_category, found_players)

    def widen_area(self):
        if self.previous_area:
            return torch.tensor([
                self.previous_area[0],
                self.previous_area[1],
                self.previous_area[2],
                self.previous_area[3]
            ])

    def center_of_area(self, coordinates):
        cor_splited = torch.split(coordinates, 1)
        center_x = (cor_splited[0] + cor_splited[2]) / 2
        center_y = (cor_splited[1] + cor_splited[3]) / 2
        return torch.cat([center_x, center_y]).unsqueeze(0)

    def define_area(self, ball, players):
        ball_x1, ball_y1, ball_x2, ball_y2, _, _ = ball.squeeze()
        area_x1, area_y1, area_x2, area_y2 = ball_x1, ball_y1, ball_x2, ball_y2

        for team, player_data in players.items():
            distance, player_tensor = player_data
            player_x1, player_y1, player_x2, player_y2, _, _ = player_tensor.squeeze()

            area_x1 = min(area_x1, player_x1)
            area_y1 = min(area_y1, player_y1)
            area_x2 = max(area_x2, player_x2)
            area_y2 = max(area_y2, player_y2)

        return torch.tensor([area_x1, area_y1, area_x2, area_y2])

    def _skirt_color(self, img, dim) -> List[float]:
        dim = torch.round(dim).squeeze(0).long()
        img = img[dim[1]:dim[3], dim[0]:dim[2]]

        bluer_img = cv2.bilateralFilter(img, 9, 100, 75)
        hsv_img = cv2.cvtColor(bluer_img, cv2.COLOR_BGR2HSV)

        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])

        green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
        non_green_mask = cv2.bitwise_not(green_mask)

        img_filtered = cv2.bitwise_and(img, img, mask=non_green_mask)

        non_zero_colors = img_filtered.reshape(-1, img_filtered.shape[-1])[
            np.any(img_filtered.reshape(-1, img_filtered.shape[-1]) != [0, 0, 0], axis=1)]

        mean_color = np.mean(non_zero_colors, axis=0)
        return list(colorsys.rgb_to_hls(*mean_color))
