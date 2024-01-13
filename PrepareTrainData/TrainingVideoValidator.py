import yolov5
import random


class TrainingVideoValidator:

    def __init__(self):
        self.model = yolov5.load('keremberke/yolov5m-football')
        self.model.conf = 0.25  # type: ignore NMS confidence threshold
        self.model.iou = 0.45  # type: ignore NMS IoU threshold
        self.model.agnostic = False  # type: ignore NMS class-agnostic
        self.model.multi_label = False  # type: ignore NMS multiple labels per box
        self.model.max_det = 1000  # type: ignore maximum number of detections per image
        self.ball_values = 0
        self.players_values = 0
        self.frames = 0

    def predict_img(self, frames):
        if len(frames) < 10:
            return False
        return True
        # self.reset_stats()
        # self.preformed_frames = 0
        # all_frames_indexies = list(range(len(frames)))
        # test_size = int(len(frames) * 0.2)
        # test_frames_indices = random.sample(all_frames_indexies, test_size)
        # for frame in test_frames_indices:
        #     results = self.model(frames[frame], size=640)
        #     self.validate_video(results)
        # return self.count_stats(len(test_frames_indices))

    def validate_video(self, pretrained_model):
        self.found_objects = pretrained_model.pred[0]
        self.frames += 1
        for element in self.found_objects:
            if element[5] == 0:
                self.ball_values += 1
            else:
                self.players_values += 1
            if self.ball_values > 0 and self.players_values > 0.:
                return

    def reset_stats(self):
        self.ball_values = 0
        self.players_values = 0
        self.preformed_frames = 0

    def count_stats(self, preformed_frames) -> bool:
        print(self.ball_values, self.players_values)
        if self.ball_values == 0 or self.players_values == 0:
            return False
        return True
