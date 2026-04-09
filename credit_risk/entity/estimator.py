import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline
from credit_risk.exception import CREDITriskException
from credit_risk.logger import logging


class CreditRiskModel:
    def __init__(self, preprocessing_object, trained_model_object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> DataFrame:
        """
        Function accepts raw inputs and transforms them using the preprocessing object
        so they are in the same format as the training data, then performs prediction
        using the trained model.
        """
        logging.info("Entered predict method of CreditRiskModel class")

        try:
            logging.info("Using the preprocessing object to transform input data")
            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Using the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise CREDITriskException(e, sys) from e
            
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"