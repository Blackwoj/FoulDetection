import yolov5


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
        for frame in frames:
            results = self.model(frame, size=640)
            self.validate_video(results)
        return self.count_stats()

    def validate_video(self, pretrained_model):
        self.found_objects = pretrained_model.pred[0]
        self.frames += 1
        for element in self.found_objects:
            if element[5] == 0:
                self.ball_values += 1
            else:
                self.players_values += 1

    def reset_stats(self):
        self.ball_values = 0
        self.players_values = 0
        self.frames = 0

    def count_stats(self) -> bool:
        if self.ball_values/self.frames < 0.5 and self.players_values/self.frames < 1.4:
            self.reset_stats()
            return False
        self.reset_stats()
        return True
