import os 
import sys 
sys.path.append("E:/ML PROJECTS/Insurance Premium Prediction")
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.componants.data_ingestion import DataIngestion



if __name__ == "__main__":
    # obj = DataIngestion()
    train_data_path,test_data_path = DataIngestion().init_data_ingestion()
   