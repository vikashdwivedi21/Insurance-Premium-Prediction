import os 
import sys 
sys.path.append("E:/ML PROJECTS/Insurance Premium Prediction")
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.componants.data_ingestion import DataIngestion
from src.componants.data_transformation import DataTransformation
from src.componants.model_trainer import ModelTrainer





if __name__ == "__main__":
    # obj = DataIngestion()
    train_data_path,test_data_path = DataIngestion().init_data_ingestion()

    data_transfromation = DataTransformation()
    train_arr, test_arr, _ = data_transfromation.init_data_transformation(train_data_path, test_data_path)

    model_trainer = ModelTrainer()
    model_trainer.init_model_training(train_arr,test_arr)
   