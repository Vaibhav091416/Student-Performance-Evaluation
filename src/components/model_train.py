#to train model on the given data
import os
import sys 
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()
    
    def initiait_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Splitting train and test data')
            x_train,y_train,x_test,y_test= (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbour Classifier":KNeighborsRegressor(),
                "XGBClassifier":XGBRegressor(),
                "CatBoosting Classifier":CatBoostRegressor(),
                "AdaBoost Classifier":AdaBoostRegressor(),
            }
            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            # it's matching name and score by index
            #list(model_report.keys())[index_of_max_score] will give the name of best model

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model found.")

            logging.info(f"Best found model on both training and testing data is {best_model_name} ".format(best_model_name=best_model_name))


            save_object(self.model_trainer_config.trained_model_file_path,best_model)

            predicted=best_model.predict(x_test)
            r2=r2_score(y_test,predicted)

            print(best_model_name)
            return r2 

        except Exception as e:
            raise CustomException(e,sys)