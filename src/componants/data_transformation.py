import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


from sklearn.impute import SimpleImputer # Handling missing values
from sklearn.preprocessing import StandardScaler # Handing feature scaling
from sklearn.preprocessing import OneHotEncoder # for encoding

#pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


## Data transformation cofig

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')



## Data Ingestion class

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def perform_data_preprocessing(self):
        try:
           logging.info("Data preprocessing has started")

           ## Define the column
           categorical_cols = ['sex', 'smoker', 'region']
           numerical_cols = ['age', 'bmi', 'children']

           logging.info("Pipeline Initiated")

           ## Numerical pipeline
           num_pipeline = Pipeline(
               steps=[
                   ('imputer',SimpleImputer(strategy='median')),
                   ('scaler',StandardScaler(with_mean=False))
               ]
           )

           ## Catagorical pipeline
           cat_pipeline = Pipeline(
               steps=[
                   ('imputer',SimpleImputer(strategy='most_frequent')),
                   ('onehotencoder',OneHotEncoder(handle_unknown='ignore')),
                   ('scaler',StandardScaler(with_mean=False))
               ]
           )

           ## Combine the pipeline
           preprocessor = ColumnTransformer([
               ('num_pipeline',num_pipeline,numerical_cols),
               ('cat_pipeline',cat_pipeline,categorical_cols)
           ])


           logging.info("Pipeline Completed")
           logging.info("Data preprocessing completed")

           return preprocessor
        
        
        except Exception as e:
            logging.info("Exception occured while performing Data Preprocessing")
            raise CustomException(e, sys)




    def init_data_transformation(self,train_path,test_path):
        
        try:
            # Read the dataset 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train Test data read completed.")
            logging.info("Printing train and test data head")
            logging.info(f"Train Dataframe Head : \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head  : \n{test_df.head().to_string()}")

            logging.info("Genrating preprocessor object")
            preprocessor = self.perform_data_preprocessing()

            target_col = 'expenses'
            drop_col = [target_col]

            input_feature_train_df = train_df.drop(columns=drop_col,axis=1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns=drop_col,axis=1)
            target_feature_test_df = test_df[target_col]

            ## Transforming data using preprocessing
            logging.info("Apply preprocessing on training and tesing dataset")
            input_feature_train_arr=preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor.transform(input_feature_test_df)

            
            
            #saving preproceesed data into numpy array for future use
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            ## Save the pickel file
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessor
            )

            logging.info('Preprocessor pickle file crated and saved')

            return( train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path,)



        except Exception as e:
            logging.info("Exception occur at Data Tranformation stage")
            raise CustomException(e,sys)