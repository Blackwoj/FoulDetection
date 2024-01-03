from .Model.DataLoader import DataLoader
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)

project_path = Path('run.py').resolve().parent
data_path = project_path / 'sampleData'

try:
    logging.info("Started!")

    data_loader = DataLoader(
        data_path,
        os.path.dirname(os.path.realpath(__file__))
    )
    data_loader._load_data()

    logging.info("Script completed successfully!")
except Exception as e:
    logging.error(f"An error occurred: {e}")
