import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton, QFileDialog
from DataProcesing.DataLoader_example import DataLoader
from DataProcesing.PrepDataToPredict import VideoDataset
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
from pathlib import Path


class DetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.pretrained_model_path = Path(__file__).resolve().parent / "tensorflow_foul_pred.h5"
        self.pretrained_model = None
        self.prep_data = VideoDataset()
        self.model = load_model(self.pretrained_model_path) if self.pretrained_model_path else None
        self.init_ui()

    def init_ui(self):
        self.detector_label = QLabel('Model detekcji faulu:', self)

        self.file_label = QLabel('Wybierz plik (tylko .mp4):', self)
        self.file_line_edit = QLineEdit(self)
        self.browse_button = QPushButton('Przeglądaj', self)
        self.browse_button.clicked.connect(self.browse_file)

        self.result_label = QLabel('Wynik detekcji:', self)

        self.run_button = QPushButton('Run', self)
        self.run_button.clicked.connect(self.run_detection)

        vbox = QVBoxLayout()
        vbox.addWidget(self.detector_label)
        vbox.addWidget(self.file_label)
        vbox.addWidget(self.file_line_edit)
        vbox.addWidget(self.browse_button)
        vbox.addWidget(self.result_label)
        vbox.addWidget(self.run_button)

        self.setLayout(vbox)

        self.setGeometry(100, 100, 400, 200)
        self.setWindowTitle('Aplikacja Detekcji')
        self.show()

    def browse_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Wybierz plik .mp4", "", "Video Files (*.mp4);;All Files (*)", options=options)

        if file_name:
            self.file_line_edit.setText(file_name)

    def run_detection(self):
        file_path = self.file_line_edit.text()

        data_loader = DataLoader()
        self.result_label.setText("Processing Danych")
        if file_path:
            processed_data = data_loader.load_data(file_path)
            result = self.detect_objects(processed_data)
            if result:
                self.result_label.setText(f'Wynik detekcji: {result}')

    def detect_objects(self, data_to_predict):
        dataset_to_predict = self.prep_data.load_data(data_to_predict)
        if dataset_to_predict:
            prediction_set = []
            for case in dataset_to_predict:
                np.size(case)
                frames_sequence = np.expand_dims(case, axis=0)
                prediction = self.model.predict(frames_sequence)  # type: ignore
                prediction_set.append(prediction[0][0])
            return "Foul" if np.mean(np.array(prediction_set)) > 0.5 else "No foul"  # type: ignore
        else:
            self.result_label.setText(f'Brak możliwości okreśelnie faulu, tylko{len(data_to_predict)} wartościowych klatek')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DetectionApp()
    sys.exit(app.exec_())
