import sys

from pandas import DataFrame

from credit_risk.entity.config_entity import CreditRiskPredictorConfig
from credit_risk.entity.s3_estimator import CreditRiskEstimator
from credit_risk.exception import CREDITriskException
from credit_risk.logger import logging


class CreditRiskData:
    def __init__(
        self,
        limit_bal,
        sex,
        education,
        marriage,
        age,
        payment_status_sep,
        payment_status_aug,
        payment_status_jul,
        payment_status_jun,
        payment_status_may,
        payment_status_apr,
        bill_statement_sep,
        bill_statement_aug,
        bill_statement_jul,
        bill_statement_jun,
        bill_statement_may,
        bill_statement_apr,
        previous_payment_sep,
        previous_payment_aug,
        previous_payment_jul,
        previous_payment_jun,
        previous_payment_may,
        previous_payment_apr,
    ):
        """
        CreditRiskData constructor
        Input: all features required by the trained model for prediction
        """
        try:
            self.limit_bal = limit_bal
            self.sex = sex
            self.education = education
            self.marriage = marriage
            self.age = age

            self.payment_status_sep = payment_status_sep
            self.payment_status_aug = payment_status_aug
            self.payment_status_jul = payment_status_jul
            self.payment_status_jun = payment_status_jun
            self.payment_status_may = payment_status_may
            self.payment_status_apr = payment_status_apr

            self.bill_statement_sep = bill_statement_sep
            self.bill_statement_aug = bill_statement_aug
            self.bill_statement_jul = bill_statement_jul
            self.bill_statement_jun = bill_statement_jun
            self.bill_statement_may = bill_statement_may
            self.bill_statement_apr = bill_statement_apr

            self.previous_payment_sep = previous_payment_sep
            self.previous_payment_aug = previous_payment_aug
            self.previous_payment_jul = previous_payment_jul
            self.previous_payment_jun = previous_payment_jun
            self.previous_payment_may = previous_payment_may
            self.previous_payment_apr = previous_payment_apr

        except Exception as e:
            raise CREDITriskException(e, sys) from e

    def get_credit_risk_input_data_frame(self) -> DataFrame:
        """
        This function returns a DataFrame from CreditRiskData class input
        """
        try:
            credit_risk_input_dict = self.get_credit_risk_data_as_dict()
            return DataFrame(credit_risk_input_dict)

        except Exception as e:
            raise CREDITriskException(e, sys) from e

    def get_credit_risk_data_as_dict(self):
        """
        This function returns a dictionary from CreditRiskData class input
        """
        logging.info("Entered get_credit_risk_data_as_dict method of CreditRiskData class")

        try:
            input_data = {
                "limit_bal": [self.limit_bal],
                "sex": [self.sex],
                "education": [self.education],
                "marriage": [self.marriage],
                "age": [self.age],
                "payment_status_sep": [self.payment_status_sep],
                "payment_status_aug": [self.payment_status_aug],
                "payment_status_jul": [self.payment_status_jul],
                "payment_status_jun": [self.payment_status_jun],
                "payment_status_may": [self.payment_status_may],
                "payment_status_apr": [self.payment_status_apr],
                "bill_statement_sep": [self.bill_statement_sep],
                "bill_statement_aug": [self.bill_statement_aug],
                "bill_statement_jul": [self.bill_statement_jul],
                "bill_statement_jun": [self.bill_statement_jun],
                "bill_statement_may": [self.bill_statement_may],
                "bill_statement_apr": [self.bill_statement_apr],
                "previous_payment_sep": [self.previous_payment_sep],
                "previous_payment_aug": [self.previous_payment_aug],
                "previous_payment_jul": [self.previous_payment_jul],
                "previous_payment_jun": [self.previous_payment_jun],
                "previous_payment_may": [self.previous_payment_may],
                "previous_payment_apr": [self.previous_payment_apr],
            }

            logging.info("Created credit risk data dictionary")
            logging.info("Exited get_credit_risk_data_as_dict method of CreditRiskData class")

            return input_data

        except Exception as e:
            raise CREDITriskException(e, sys) from e


class CreditRiskClassifier:
    def __init__(
        self,
        prediction_pipeline_config: CreditRiskPredictorConfig = CreditRiskPredictorConfig(),
    ) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise CREDITriskException(e, sys) from e

    def predict(self, dataframe) -> str:
        """
        This is the method of CreditRiskClassifier
        Returns: Prediction output
        """
        try:
            logging.info("Entered predict method of CreditRiskClassifier class")

            model = CreditRiskEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )

            result = model.predict(dataframe)

            return result

        except Exception as e:
            raise CREDITriskException(e, sys) from e