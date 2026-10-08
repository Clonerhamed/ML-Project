import sys

from ml_project.exception import CustomException
from ml_project.logger import logging

from ml_project.components.data_ingestion import DataIngestion
from ml_project.components.data_transformation import DataTransformation
from ml_project.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")

            # =========================================================
            # Step 1: Data Ingestion
            # =========================================================
            logging.info("Starting data ingestion")

            data_ingestion = DataIngestion()

            train_data_path, test_data_path = (
                data_ingestion.initiate_data_ingestion()
            )

            logging.info(
                f"Data ingestion completed. "
                f"Train path: {train_data_path}, "
                f"Test path: {test_data_path}"
            )

            # =========================================================
            # Step 2: Data Transformation
            # =========================================================
            logging.info("Starting data transformation")

            data_transformation = DataTransformation()

            train_arr, test_arr, preprocessor_path = (
                data_transformation.initiate_data_transformation(
                    train_data_path,
                    test_data_path
                )
            )

            logging.info(
                f"Data transformation completed. "
                f"Preprocessor saved at: {preprocessor_path}"
            )

            # =========================================================
            # Step 3: Model Training
            # =========================================================
            logging.info("Starting model training")

            model_trainer = ModelTrainer()

            model_score = model_trainer.initiate_model_trainer(
                train_arr,
                test_arr
            )

            logging.info(
                f"Model training completed. "
                f"Final R2 Score: {model_score}"
            )

            logging.info("Training pipeline completed successfully")

            return model_score

        except Exception as e:
            logging.error("Error occurred in training pipeline")
            raise CustomException(e, sys)


if __name__ == "__main__":

    train_pipeline = TrainPipeline()

    score = train_pipeline.run_pipeline()

    print("\n===================================")
    print("Training completed successfully")
    print(f"Final R2 Score: {score:.4f}")
    print("===================================")