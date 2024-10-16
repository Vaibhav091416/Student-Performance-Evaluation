#for data ingestion e.g. reding from a database or so
import os
import sys 
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

class DataIngstionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')  # "train_data_path:str" is a type hint telling train_data_path expects a string. this syntax is called annotations
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngstionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data Ingestion method or Component")
        try:
            df=pd.read_csv('.//notebook//data//stud.csv')
            logging.info("Read the data as DataFrame")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) 
            #os.path.dirname returns the path saved in ingestion_config.train_path
            #os.makedir() now having the path from os.path.dirname() will create the directories
            #basically creates file called artifacts
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            #saves file in the created directory
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=43)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of Data Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    obj=DataIngestion()
    obj.initiate_data_ingestion()