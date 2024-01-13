import os
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

class VideoDataset(Dataset):
    def __init__(self, root_dir: Path, num_frames_per_sample=10):
        self.root_dir = root_dir
        self.num_frames_per_sample = num_frames_per_sample
        self.data = self.load_data()

    def load_data(self):
        data = []

        for label in os.listdir(self.root_dir):
            label_folder = self.root_dir / label
            if os.path.isdir(label_folder):
                for csv_file in label_folder.glob('*.csv'):
                    df = pd.read_csv(csv_file)
                    if len(df) >= self.num_frames_per_sample:
                        self.data_multi = len(df) // self.num_frames_per_sample
                        if self.data_multi == 1:
                            data.append((df.iloc[-10:, :], label))
                        else:
                            for i in range(self.data_multi):
                                sequence = []
                                j = i
                                while j < len(df):
                                    sequence.append(df.iloc[j, :])
                                    j += self.data_multi
                                data.append((sequence, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, label = self.data[idx]
        # Tutaj możesz przetwarzać sekwencję, np. przy użyciu Pandas DataFrame lub konwertować na tensor
        # Zwróć tuple (sekwencja, label)