import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from Model.DataCsvReader import VideoDataset
import torchvision.models as models


class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomModel, self).__init__()

        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.resnet(x)

        batch_size, seq_length, c, h, w = x.size()
        x = x.view(batch_size, seq_length, -1)
        _, (x, _) = self.lstm(x)

        x = self.fc(x[-1, :, :])

        return x


def train(model, data, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0
        for input, target in data:
            optimizer.zero_grad()

            output = model(input)

            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data)}")


def main():
    train_data_folder = Path('C:/Users/wnikiel/Documents/FOuldetectionProject/FoulDetection/trainData')
    video_dataset = VideoDataset(root_dir=train_data_folder, num_frames_per_sample=10)

    data = video_dataset.load_data()

    input_size = 103
    hidden_size = 128
    num_classes = 1

    model = CustomModel(input_size, hidden_size, num_classes)

    epochs = 10
    learning_rate = 0.001

    train(model, data, epochs, learning_rate)


if __name__ == '__main__':
    main()
