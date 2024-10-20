#non-numerical to numerical etc
import os 
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_obj_file_path=os.path.join("artifacts",'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig() ##
    def get_transformer_obj(self):
        '''
        Data Transformation function.
        '''
        try:
            numerical_columns=["reading_score","writing_score"]
            categorical_column=[
                "gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"
            ]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),   #handling missing value
                    ('scaler',StandardScaler())                     #standard scaler
                ]
            )

            cat_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='most_frequent')),
                ("one_hot_encoder",OneHotEncoder()),
                # ("scaling",StandardScaler())
            ])

            logging.info(f"Numerical Columns: {numerical_columns}")
            logging.info(f"Categorical Columns:{categorical_column}")

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('categorical_pipeline',cat_pipeline,categorical_column)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Train and Test Data Compleated")

            logging.info("Obtaining preprocessing Object")

            preprocessor_obj=self.get_transformer_obj()

            target_column='math_score'
            numerical_columns=['writing_score','reading_score']

            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info(f"Applying preprocessing Object on train da taframe and test dataframe")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)      ##fit transform
            input_feature_test_arr=preprocessor_obj.fit_transform(input_feature_test_df)

            train_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info(f"Saved Preprocessing Object.")


            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessor_obj
                        )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)

