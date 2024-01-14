import os
import pandas as pd
from pathlib import Path
import torch


class VideoDataset:
    def __init__(self, root_dir: Path, num_frames_per_sample=10):
        self.root_dir = root_dir
        self.num_frames_per_sample = num_frames_per_sample

    def load_data(self):
        data = []

        for label in os.listdir(self.root_dir):
            label_folder = self.root_dir / label
            prob = 1.0 if label == "Foul" else 0.0
            if os.path.isdir(label_folder):
                for csv_file in label_folder.glob('*.csv'):
                    df = pd.read_csv(csv_file, header=None)
                    if len(df) >= self.num_frames_per_sample:
                        self.data_multi = len(df) // self.num_frames_per_sample
                        if self.data_multi == 1:
                            data.append((torch.tensor(df.iloc[-10:, :].values).double(), prob))
                        else:
                            for i in range(self.data_multi):
                                sequence = []
                                j = i
                                while j < len(df) and len(sequence) < 10:
                                    sequence.append(df.iloc[j, :].values)
                                    j += self.data_multi
                                data.append((torch.tensor(sequence).double(), prob))
        return data
