import sys
import numpy as np
import pandas as pd

from imblearn.combine import SMOTEENN
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer

from credit_risk.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from credit_risk.entity.config_entity import DataTransformationConfig
from credit_risk.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
)
from credit_risk.exception import CREDITriskException
from credit_risk.logger import logging
from credit_risk.utils.main_utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file,
)


class CreditRiskFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Custom transformer that applies the same cleaning and encoding logic
    used in the notebook, so the pipeline stays consistent with the models that were already tested.
    """

    def __init__(self, drop_columns=None):
        self.drop_columns = drop_columns if drop_columns is not None else []

        self.sex_map = {
            "Female": 0,
            "Male": 1
        }

        self.education_map = {
            "Others": 0,
            "High school": 1,
            "University": 2,
            "Graduate school": 3
        }

        self.marriage_map = {
            "Single": 0,
            "Married": 1,
            "Others": 2
        }

        self.payment_cols = [
            "payment_status_sep",
            "payment_status_aug",
            "payment_status_jul",
            "payment_status_jun",
            "payment_status_may",
            "payment_status_apr"
        ]

    def fit(self, X, y=None):
        return self

    def _map_payment_status(self, value):
        if pd.isna(value):
            return 0
        if value in ["Payed duly", "Unknown"]:
            return 0
        if isinstance(value, str) and "Payment delayed" in value:
            return int(value.split()[2])
        return value

    def transform(self, X):
        df = X.copy()

        #drop unused columns
        for col in self.drop_columns:
            if col in df.columns:
                df = df.drop(columns=[col])

        #replace string "na" with actual NaN
        df.replace({"na": np.nan}, inplace=True)

        #fill missing values
        if "age" in df.columns:
            df["age"] = df["age"].fillna(df["age"].median())

        for col in ["sex", "education", "marriage"]:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])

        #encode categorical variables
        if "sex" in df.columns:
            df["sex"] = df["sex"].map(self.sex_map)

        if "education" in df.columns:
            df["education"] = df["education"].map(self.education_map)

        if "marriage" in df.columns:
            df["marriage"] = df["marriage"].map(self.marriage_map)

        #encode payment status columns
        for col in self.payment_cols:
            if col in df.columns:
                df[col] = df[col].apply(self._map_payment_status)

        return df


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CREDITriskException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CREDITriskException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates the full preprocessing pipeline.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            drop_columns = self._schema_config["drop_columns"]
            transform_columns = self._schema_config["transform_columns"]
            num_features = self._schema_config["num_features"]

            #scale columns
            remaining_numeric_features = [
                col for col in num_features if col not in transform_columns
            ]

            feature_builder = CreditRiskFeatureBuilder(drop_columns=drop_columns)

            power_scale_pipeline = Pipeline(
                steps=[
                    ("power_transform", PowerTransformer(method="yeo-johnson")),
                    ("scaler", StandardScaler())
                ]
            )

            scale_only_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            #these are already encoded by the custom transformer
            encoded_columns = [
                "sex",
                "education",
                "marriage",
                "payment_status_sep",
                "payment_status_aug",
                "payment_status_jul",
                "payment_status_jun",
                "payment_status_may",
                "payment_status_apr"
            ]

            column_transformer = ColumnTransformer(
                transformers=[
                    ("power_scale", power_scale_pipeline, transform_columns),
                    ("scale_only", scale_only_pipeline, remaining_numeric_features),
                    ("encoded_passthrough", "passthrough", encoded_columns),
                ],
                remainder="drop",
                verbose_feature_names_out=False
            )

            preprocessor = Pipeline(
                steps=[
                    ("feature_builder", feature_builder),
                    ("column_transformer", column_transformer)
                ]
            )

            logging.info("Created preprocessing pipeline successfully")
            return preprocessor

        except Exception as e:
            raise CREDITriskException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates data transformation for the pipeline.
        """
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            logging.info("Starting data transformation")

            preprocessor = self.get_data_transformer_object()
            logging.info("Got preprocessing object")

            train_df = DataTransformation.read_data(
                file_path=self.data_ingestion_artifact.trained_file_path
            )
            test_df = DataTransformation.read_data(
                file_path=self.data_ingestion_artifact.test_file_path
            )

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info("Separated input and target features for train and test datasets")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Applied preprocessing pipeline to training and testing features")

            #apply SMOTEENN only to training data
            smt = SMOTEENN(sampling_strategy="minority", random_state=42)

            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr,
                target_feature_train_df
            )

            logging.info("Applied SMOTEENN to training dataset only")

            #balancing train and test data
            input_feature_test_final = input_feature_test_arr
            target_feature_test_final = target_feature_test_df.to_numpy()

            train_arr = np.c_[
                input_feature_train_final,
                np.array(target_feature_train_final)
            ]

            test_arr = np.c_[
                input_feature_test_final,
                np.array(target_feature_test_final)
            ]

            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            logging.info("Saved preprocessor object and transformed train/test arrays")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise CREDITriskException(e, sys) from e