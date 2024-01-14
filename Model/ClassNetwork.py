import torch.nn as nn

class ResNetConvLSTM(nn.Module):
    def __init__(self, resnet_model, input_size, hidden_size, num_classes, lstm_kernel_size=3):
        super(ResNetConvLSTM, self).__init__()
        self.resnet = resnet_model
        self.resnet.conv1 = nn.Conv2d(103, 64, kernel_size=7, stride=2, padding=3, bias=False).double()
        self.resnet.fc = nn.Identity()

        for layer in self.resnet.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Linear):
                layer = layer.double()

        self.hidden_size = hidden_size
        self.lstm_kernel_size = lstm_kernel_size
        self.num_features = input_size[2]  # Wybierz liczbę cech (third element in input_size)
        self.batch_size = None  # Rozmiar batcha będzie ustawiany przy pierwszym przejściu przez sieć

        self.lstm = None  # Ukryta warstwa LSTM nie jest inicjalizowana tutaj
        self.fc = nn.Linear(hidden_size, num_classes).double()
        self.hidden = None  # Ukryta warstwa LSTM nie jest inicjalizowana tutaj

    def init_hidden(self, batch_size):
        # Inicjalizuj ukrytą warstwę LSTM
        weight = next(self.parameters()).data
        return (weight.new(1, batch_size, self.hidden_size).zero_(),
                weight.new(1, batch_size, self.hidden_size).zero_())

    def forward(self, x):
        batch_size, seq_len, num_features = x.size()
        print(batch_size, seq_len, num_features)

        if self.batch_size is None:
            # Inicjalizuj rozmiar batcha i ukrytą warstwę LSTM przed pierwszym przejściem
            self.batch_size = batch_size
            self.lstm = nn.LSTM(
                input_size=num_features,
                hidden_size=int(self.hidden_size),
                num_layers=1,
                batch_first=True
            ).double()
            self.hidden = self.init_hidden(self.batch_size)

        features = self.resnet(x.view(batch_size * seq_len, num_features, 1, 1))
        print(features.size())
        features = features.view(batch_size, seq_len, -1)

        # LSTM input should have the shape (batch_size, seq_len, input_size)
        print(features.size())
        lstm_input = features.permute(0, 2, 1).contiguous()
        print(lstm_input.size())

        # Upewnij się, że tensor wejściowy do LSTM ma odpowiednie wymiary
        print(f'LSTM input size: {lstm_input.size()}')
        print(f'LSTM hidden size: {self.hidden_size}')

        # Dopasuj wymiary tensora wejściowego do warstwy LSTM
        lstm_input = lstm_input.view(batch_size, self.hidden_size, seq_len)

        h_t, self.hidden = self.lstm(lstm_input, self.hidden)

        # Pozostały kod bez zmian

        # Use the last time step output for classification
        h_t = h_t[:, -1, :]

        # Classification
        output = self.fc(h_t)
        return output
