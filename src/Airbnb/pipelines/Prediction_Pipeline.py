import os
import numpy as np
import sys
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any
from src.Airbnb.logger import logging
from src.Airbnb.utils.utils import load_object
from src.Airbnb.exception import customexception


class PredictPipeline:
    def __init__(self):
        """
        Initialize the prediction pipeline.
        This constructor loads the preprocessor and model paths for later use.
        """
        self.preprocessor_path = os.path.join("Artifacts", "Preprocessor.pkl")
        self.model_path = os.path.join("Artifacts", "Model.pkl")
    
    def predict(self, features):
        """
        Make predictions on the provided features.
        
        Args:
            features: Input features as a pandas DataFrame
            
        Returns:
            numpy.ndarray: Predicted values
        """
        try:
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)
            logging.info('Preprocessor and Model Pickle files loaded')
            scaled_data = preprocessor.transform(features)
            logging.info('Data Scaled')
            pred = model.predict(scaled_data)
            return pred
        except Exception as e:
            raise customexception(e, sys)


@dataclass
class PropertyFeatures:
    """Data class to store all Airbnb property features."""
    property_type: str
    room_type: str
    amenities: int
    accommodates: int
    bathrooms: int
    bed_type: str
    cancellation_policy: str
    cleaning_fee: float
    city: str
    host_has_profile_pic: str
    host_identity_verified: str
    host_response_rate: str
    instant_bookable: str
    latitude: float
    longitude: float
    number_of_reviews: int
    review_scores_rating: int
    bedrooms: int
    beds: int


class CustomData:
    def __init__(self, features: PropertyFeatures):
        """
        Initialize CustomData with property features.
        
        Args:
            features: PropertyFeatures dataclass containing all property attributes
        """
        self.features = features

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Create CustomData instance from a dictionary.
        
        Args:
            data: Dictionary containing all property features
            
        Returns:
            CustomData: New instance with features from dictionary
        """
        features = PropertyFeatures(**data)
        return cls(features)

    def get_data_as_dataframe(self):
        """
        Convert the custom data instance into a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with a single row containing all property features
            
        Raises:
            customexception: If any error occurs during DataFrame creation
        """
        try:
            custom_data_input_dict = {
                'property_type': [self.features.property_type],
                'room_type': [self.features.room_type],
                'amenities': [self.features.amenities],
                'accommodates': [self.features.accommodates],
                'bathrooms': [self.features.bathrooms],
                'bed_type': [self.features.bed_type],
                'cancellation_policy': [self.features.cancellation_policy],
                'cleaning_fee': [self.features.cleaning_fee],
                'city': [self.features.city],
                'host_has_profile_pic': [self.features.host_has_profile_pic],
                'host_identity_verified': [self.features.host_identity_verified],
                'host_response_rate': [self.features.host_response_rate],
                'instant_bookable': [self.features.instant_bookable],
                'latitude': [self.features.latitude],
                'longitude': [self.features.longitude],
                'number_of_reviews': [self.features.number_of_reviews],
                'review_scores_rating': [self.features.review_scores_rating],
                'bedrooms': [self.features.bedrooms],
                'beds': [self.features.beds]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occurred in prediction pipeline')
            raise customexception(e, sys)