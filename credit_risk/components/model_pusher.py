import sys

from credit_risk.cloud_storage.aws_storage import SimpleStorageService
from credit_risk.exception import CREDITriskException
from credit_risk.logger import logging
from credit_risk.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from credit_risk.entity.config_entity import ModelPusherConfig
from credit_risk.entity.s3_estimator import CreditRiskEstimator


class ModelPusher:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig
    ):
        """
        :param model_evaluation_artifact: Output reference of model evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.credit_risk_estimator = CreditRiskEstimator(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.s3_model_key_path
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   This function is used to initiate all steps of the model pusher

        Output      :   Returns model pusher artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            logging.info("Uploading trained model to S3 bucket")

            self.credit_risk_estimator.save_model(
                from_file=self.model_evaluation_artifact.trained_model_path
            )

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path
            )

            logging.info("Uploaded trained model to S3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except Exception as e:
            raise CREDITriskException(e, sys) from e