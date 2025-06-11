import os
from pathlib import Path
import pandas as pd


def normalize_position(df, width, height):
    df[99] = df[99] / width
    df[100] = df[100] / height
    df[101] = df[101] / width
    df[102] = df[102] / height
    return df


def process_and_save_csv(input_path, output_path, width, height):
    df = pd.read_csv(input_path, header=None)

    df_normalized = normalize_position(df, width, height)

    df_normalized.to_csv(output_path, index=False, header=None)


def main():
    directory_path = Path(__file__).resolve().parent
    directory_path = directory_path / 'trainData' / 'NoFoul'

    image_width = 854
    image_height = 480

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            input_file = os.path.join(directory_path, filename)
            output_file = os.path.join(directory_path, f"normalized_{filename}")

            process_and_save_csv(input_file, output_file, image_width, image_height)

if __name__ == "__main__":
    main()
