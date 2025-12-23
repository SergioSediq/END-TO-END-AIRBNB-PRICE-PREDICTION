import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.Airbnb.logger import logging
from catboost import CatBoostRegressor
from src.Airbnb.utils.utils import save_object
from src.Airbnb.exception import customexception
from src.Airbnb.utils.utils import evaluate_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('Artifacts','Model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(random_state=42, alpha=1.0),
                'Ridge': Ridge(random_state=42, alpha=1.0),
                'Elasticnet': ElasticNet(random_state=42, alpha=1.0, l1_ratio=0.5),
                'RandomForestRegressor': RandomForestRegressor(
                    random_state=42,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    n_estimators=100
                ),
                'GradientBoostingRegressor': GradientBoostingRegressor(
                    random_state=42,
                    learning_rate=0.1,
                    n_estimators=100
                ),
                'CatBoostRegressor': CatBoostRegressor(
                    random_state=42,
                    verbose=False,
                    iterations=100,
                    learning_rate=0.1,
                    depth=6
                )
            }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
          
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)