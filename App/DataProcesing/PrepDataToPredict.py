import pandas as pd
import torch


class VideoDataset:
    def __init__(self):
        self.num_frames_per_sample = 10

    def load_data(self, data_to_process):
        data = []
        df = pd.DataFrame(data_to_process)
        if len(df) >= self.num_frames_per_sample:
            self.data_multi = len(df) // self.num_frames_per_sample
            if self.data_multi == 1:
                data.append(torch.tensor(df.iloc[-10:, :].values).double())
            else:
                for i in range(self.data_multi):
                    sequence = []
                    j = i
                    while j < len(df) and len(sequence) < 10:
                        sequence.append(df.iloc[j, :].values)
                        j += self.data_multi
                    data.append(torch.tensor(sequence).double())
        return data
