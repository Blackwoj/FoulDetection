from .Model.DataCsvReader import VideoDataset
from .Model.ClassNetwork import TimeSeriesResNet
from pathlib import Path
from torch.utils.data import DataLoader

# Ścieżka do folderu zawierającego dane treningowe (Foul i NoFoul)
train_data_folder = Path('C:/Users/wnikiel/Documents/FOuldetectionProject/FoulDetection/trainData')

# Tworzenie instancji VideoDataset
video_dataset = VideoDataset(root_dir=train_data_folder, num_frames_per_sample=10)

# Tworzenie DataLoader z VideoDataset
train_loader = DataLoader(video_dataset, batch_size=64, shuffle=True, num_workers=4)

# Tworzenie instancji TimeSeriesResNet
model = TimeSeriesResNet(input_size=3, output_size=10, num_frames_per_sample=10, learning_rate=0.001)

# Trening modelu na danych z DataLoader
model.train(train_loader, epochs=10)

# Zapisanie wytrenowanego modelu
model.save_model("model.pth")
