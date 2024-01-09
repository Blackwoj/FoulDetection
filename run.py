import logging
import os
from pathlib import Path
from Model.DataLoader import DataLoader as MyDataLoader


logging.basicConfig(level=logging.INFO)

project_path = Path('run.py').resolve().parent
data_path = project_path / 'sampleData'

try:
    logging.info("Started!")

    data_loader = MyDataLoader(
        data_path,
        os.path.dirname(os.path.realpath(__file__))
    )
    logging.info("Loading data.")
    data_loader._load_data()

    logging.info("Script completed successfully!")
except Exception as e:
    logging.error(f"An error occurred: {e}")
