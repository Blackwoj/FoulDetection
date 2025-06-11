import torch
from pathlib import Path
from Model.DataCsvReader import VideoDataset
from tensorflow.keras import layers, models
import numpy as np


def create_tensorflow_model(input_shape):
    model = models.Sequential()
    model.add(layers.LSTM(50, input_shape=input_shape, return_sequences=True))
    model.add(layers.LSTM(50, return_sequences=True))
    model.add(layers.LSTM(50))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


train_data_folder = Path('C:/Users/wnikiel/Documents/FOuldetectionProject/FoulDetection/trainData')
video_dataset = VideoDataset(root_dir=train_data_folder, num_frames_per_sample=10)

data = video_dataset.load_data()

all_data = []
for record in data:
    frames = record[0]
    label = torch.tensor(record[-1], dtype=torch.double)
    all_data.append((frames, label))


train_size = int(0.8 * len(all_data))
test_size = len(all_data) - train_size
train_data, test_data = torch.utils.data.random_split(all_data, [train_size, test_size])

input_shape = (10, 103)
tensorflow_model = create_tensorflow_model(input_shape)
batch_size = 32

x_train = np.array([item[0].numpy() for item in train_data])
y_train = np.array([item[1].numpy() for item in train_data])
tensorflow_model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_split=0.2)

tensorflow_model.save('tensorflow_foul_pred.h5')

x_test = np.array([item[0].numpy() for item in test_data])
y_test = np.array([item[1].numpy() for item in test_data])

test_loss, test_accuracy = tensorflow_model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
