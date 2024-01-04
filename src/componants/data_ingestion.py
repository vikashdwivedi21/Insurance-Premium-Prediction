import os 
import sys 
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

# Initializing data ingestion config
@dataclass
class DataIngestionConfig:
    ##create path for the train test split
    raw_data_path = os.path.join('artifacts','raw.csv')   ## raw file contain orignal unsplitted dataset 
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')


## Class for data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def init_data_ingestion(self):
        logging.info("Data Ingestion has started.")
       
        try:

            ## Read the dataset 
            df = pd.read_csv('notebook\data\insurance.csv')
            logging.info("Reading of the dataset is completed")

            # save the data in raw file before spliting as train and test data 
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index= False)

            ## Split the data into train and test
            logging.info("Splitting data into train test set")
            train_set, test_set = train_test_split(df,test_size=0.20,random_state=30)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) 
            
            logging.info("Data ingestion is completed")

            return(self.ingestion_config.train_data_path,self.ingestion_config.test_data_path )


        except Exception as e:
            logging.info("Exception occured at the Data Ingesgtion stage.")
            raise CustomException(e, sys)


