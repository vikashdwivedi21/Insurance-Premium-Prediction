import os
import sys
import numpy as np
import pandas as pd
import pickle

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

## Function for create and save the pickel file
def save_object(file_path,obj):
    try:
        logging.info("Dumping object into pickel file")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        logging.info("Exception is occur in pickel file creation ")
        raise CustomException(e,sys)


## function for model evaluation
def evaluate_model(X_train,y_train,X_test,y_test,models):

    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]

            ## Traing the model
            logging.info("model traing starts")
            model.fit(X_train,y_train)

            ## Predict Testing data
            y_pred = model.predict(X_test)

            # Evaluate model and get r2_score
            test_model_score = r2_score(y_test,y_pred)

            report[list(models.keys())[i]] = test_model_score
            logging.info("Model Evaluation completed.")
            return report


    except Exception as e:
        logging.info("Exception is occur in model evaluation ")
        raise CustomException(e,sys)
    


# Function for loading the object  in pickel file
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
         


    except Exception as e:
        logging.info('Exception occured in load object function stage in utils file')
        raise CustomException(e,sys)