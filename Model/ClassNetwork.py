import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


class TimeSeriesResNet:
    def __init__(self, input_size, output_size, num_frames_per_sample, learning_rate=0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.num_frames_per_sample = num_frames_per_sample
        self.learning_rate = learning_rate

        # Tworzenie modelu ResNet
        self.model = self._build_model()

        # Definiowanie funkcji straty i optymalizatora
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def _build_model(self):
        # Tworzenie modelu ResNet18
        resnet = models.resnet18(pretrained=True)

        # Zamiana ostatniej warstwy (klasyfikatora) na nową warstwę dla naszego przypadku
        resnet.fc = nn.Linear(resnet.fc.in_features, self.output_size)

        # Replicowanie warstw dla num_frames_per_sample
        model = nn.Sequential(
            nn.Conv2d(self.input_size * self.num_frames_per_sample, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet,
            nn.Flatten(),
            nn.Linear(self.output_size, self.output_size)
        )

        return model

    def train(self, train_loader, epochs=10):
        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0

            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

# input_size - rozmiar wejścia (np. liczba cech na ramkę)
# output_size - liczba klas
# num_frames_per_sample - liczba klatek w sekwencji czasowej
# learning_rate - współczynnik uczenia
# train_loader - DataLoader z danymi uczącymi
# epochs - liczba epok
# model_path - ścieżka do zapisania modelu
# Przykład użycia:
# model = TimeSeriesResNet(input_size=3, output_size=10, num_frames_per_sample=10, learning_rate=0.001)
# model.train(train_loader, epochs=10)
# model.save_model("model.pth")
