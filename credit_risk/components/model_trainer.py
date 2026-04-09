import sys
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from neuro_mf import ModelFactory

from credit_risk.exception import CREDITriskException
from credit_risk.logger import logging
from credit_risk.utils.main_utils import load_numpy_array_data, load_object, save_object
from credit_risk.entity.config_entity import ModelTrainerConfig
from credit_risk.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)
from credit_risk.entity.estimator import CreditRiskModel


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):
        """
        :param data_transformation_artifact: Output reference of data transformation stage
        :param model_trainer_config: Configuration for model trainer
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(
        self,
        train: np.array,
        test: np.array
    ) -> Tuple[object, ClassificationMetricArtifact]:
        """
        Method Name :   get_model_object_and_report
        Description :   Uses neuro_mf to get the best model object and evaluation report

        Output      :   Returns best model details and metric artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Using neuro_mf to get best model object and report")

            model_factory = ModelFactory(
                model_config_path=self.model_trainer_config.model_config_file_path
            )

            x_train, y_train = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]

            best_model_detail = model_factory.get_best_model(
                X=x_train,
                y=y_train,
                base_accuracy=self.model_trainer_config.expected_accuracy
            )

            model_obj = best_model_detail.best_model
            y_pred = model_obj.predict(x_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            logging.info(f"Model Accuracy: {accuracy}")
            logging.info(f"Model F1 Score: {f1}")
            logging.info(f"Model Precision: {precision}")
            logging.info(f"Model Recall: {recall}")

            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1,
                precision_score=precision,
                recall_score=recall
            )

            return best_model_detail, metric_artifact

        except Exception as e:
            raise CREDITriskException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model trainer steps

        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )

            best_model_detail, metric_artifact = self.get_model_object_and_report(
                train=train_arr,
                test=test_arr
            )

            preprocessing_obj = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")

            credit_risk_model = CreditRiskModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model_detail.best_model
            )

            logging.info("Created CreditRiskModel object with preprocessor and trained model")
            logging.info("Saving trained model object")

            save_object(
                self.model_trainer_config.trained_model_file_path,
                credit_risk_model
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise CREDITriskException(e, sys) from e