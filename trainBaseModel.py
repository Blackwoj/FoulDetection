import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from Model.DataCsvReader import VideoDataset
import torchvision.models as models


class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomModel, self).__init__()

        # ResNet
        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # ResNet forward
        x = self.resnet(x)

        # LSTM forward
        # Assuming input size is (batch_size, sequence_length, features)
        batch_size, seq_length, c, h, w = x.size()
        x = x.view(batch_size, seq_length, -1)
        _, (x, _) = self.lstm(x)

        # Fully connected layer forward
        x = self.fc(x[-1, :, :])

        return x


def train(model, data, epochs, learning_rate):
    criterion = nn.MSELoss()  # Funkcja straty (Mean Squared Error)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Optymalizator (Adam)

    for epoch in range(epochs):
        total_loss = 0.0
        for input, target in data:
            # Wyczyszczenie gradientów
            optimizer.zero_grad()

            # Przekazanie danych przez model
            output = model(input)

            # Obliczenie straty
            loss = criterion(output, target)

            # Wsteczna propagacja i aktualizacja wag
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Wydruk straty po każdej epoce
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data)}")


def main():
    # Tworzenie instancji VideoDataset
    train_data_folder = Path('C:/Users/wnikiel/Documents/FOuldetectionProject/FoulDetection/trainData')
    video_dataset = VideoDataset(root_dir=train_data_folder, num_frames_per_sample=10)

    # Wczytanie danych bez użycia DataLoader
    data = video_dataset.load_data()

    # Przykładowe ustawienia
    input_size = 103
    hidden_size = 128
    num_classes = 1

    # Inicjalizacja modelu
    model = CustomModel(input_size, hidden_size, num_classes)

    # Przykładowe ustawienia treningowe
    epochs = 10
    learning_rate = 0.001

    # Rozpoczęcie procesu uczenia
    train(model, data, epochs, learning_rate)


if __name__ == '__main__':
    main()
