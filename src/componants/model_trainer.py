import os
import sys
import pandas as pd 
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    
    def init_model_training(self,train_arr,test_arr):

        try:
            logging.info("Splitting dependent and independent variable from train and test arr ")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
        
            models = {
                'Adaboost': AdaBoostRegressor(learning_rate=0.01,n_estimators=100),
                'Random_forest': RandomForestRegressor(max_depth=10, max_features='sqrt',
                                                       min_samples_leaf=1, min_samples_split=5,
                                                       n_estimators=50)

            }

            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n=================================================================================================')
            logging.info(f'model Report:{model_report}')

            # To get best model from the model dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best model found, Model name :{best_model_name}, R2 Score: {best_model_score} ')
            print('\n=============================================================================================================')
            logging.info(f'Best model found, Model name :{best_model_name}, R2 Score: {best_model_score} ')

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )


        except Exception as e:
            logging.info('Exception occure at model training stage')
            raise CustomException(e,sys)