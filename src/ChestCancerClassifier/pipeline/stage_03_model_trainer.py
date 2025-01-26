# pipeline_stage.py

import sys
import os
import logging
from pathlib import Path

# Allow import from src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ChestCancerClassifier.config.configuration import ConfigurationManager
from ChestCancerClassifier.components.model_trainer import Training
from ChestCancerClassifier import logger  # Ensure this is configured properly

# DagsHub + MLflow imports
import dagshub
import mlflow
import mlflow.tensorflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info("Loading configuration...")
        # 1) Load configuration
        config = ConfigurationManager()
        training_config = config.get_training_config()

        logger.info("Initializing Training component...")
        # 2) Initialize Training component
        training = Training(config=training_config)

        # 3) Initialize DagsHub MLflow integration using only the repository URI
        # Ensure that your environment is authenticated with DagsHub via SSH or Git credentials
        dagshub.init(
            repo_owner="Annmary777",      # Replace with your DagsHub username
            repo_name="Fall-Detection",   # Replace with your DagsHub repo name
            mlflow=True
        )

        logger.info("Starting MLflow run...")
        # 4) Start an MLflow run
        with mlflow.start_run(run_name=STAGE_NAME):
            logger.info("Logging training parameters...")
            # (Optional) Log training_config parameters automatically, if they start with 'params_'
            for attr_name in dir(training_config):
                if attr_name.startswith("params_"):
                    value = getattr(training_config, attr_name)
                    mlflow.log_param(attr_name, value)

            logger.info("Executing training steps...")
            # 5) Execute standard training steps
            training.get_base_model()
            training.train_valid_generator()

            logger.info("Training the model...")
            # 6) Train the model and capture metrics
            metrics = training.train()

            logger.info("Logging metrics...")
            # 7) Log metrics properly
            if metrics and isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    # If the value is an int/float, log as metric
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
                        logger.info(f"Logged metric: {metric_name} = {metric_value}")
                    else:
                        # Otherwise, log it as a param (string) to avoid INVALID_PARAMETER_VALUE
                        mlflow.log_param(metric_name, str(metric_value))
                        logger.info(f"Logged param: {metric_name} = {metric_value}")

            logger.info("Logging the trained model to MLflow...")
            # 8) Log the trained model
            mlflow.tensorflow.log_model(
                model=training.model,  # Correct keyword argument
                artifact_path="model",
                registered_model_name="ChestCancerModel"  # Optional: Register the model in MLflow Model Registry
            )
            logger.info("Model logged successfully.")

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
