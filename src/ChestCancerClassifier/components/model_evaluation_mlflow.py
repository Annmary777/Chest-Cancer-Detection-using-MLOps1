# ChestCancerClassifier/components/model_evaluation_mlflow.py

import tensorflow as tf
from pathlib import Path
from ChestCancerClassifier.entity.config_entity import EvaluationConfig
from ChestCancerClassifier.utils.common import save_json
import logging
from urllib.parse import urlparse
import os

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.score = None
        self.model = None
        self.valid_generator = None
        self.logger = logging.getLogger(__name__)

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1. / 255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self) -> dict:
        self.logger.info("Loading the trained model...")
        self.model = self.load_model(self.config.path_of_model)
        self.logger.info("Preparing validation data generator...")
        self._valid_generator()
        self.logger.info("Evaluating the model on validation data...")
        self.score = self.model.evaluate(self.valid_generator, verbose=0)
        self.logger.info(f"Evaluation scores: Loss = {self.score[0]}, Accuracy = {self.score[1]}")
        results = {"loss": self.score[0], "accuracy": self.score[1]}
        incorrect_predictions = self.get_incorrect_predictions()
        num_incorrect = len(incorrect_predictions)
        self.logger.info(f"Number of incorrect predictions: {num_incorrect}")
        self.save_score()
        self.save_incorrect_predictions(incorrect_predictions)
        results["num_incorrect"] = num_incorrect
        return results

    def get_incorrect_predictions(self) -> list:
        self.logger.info("Generating predictions for validation data...")
        predictions = self.model.predict(self.valid_generator, verbose=0)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        true_classes = self.valid_generator.classes
        class_indices = self.valid_generator.class_indices
        class_labels = {v: k for k, v in class_indices.items()}

        # Identify incorrect predictions
        incorrect_indices = [i for i, (pred, true) in enumerate(zip(predicted_classes, true_classes)) if pred != true]
        self.logger.info(f"Found {len(incorrect_indices)} incorrect predictions.")

        # Get filenames of incorrect predictions
        incorrect_filenames = [self.valid_generator.filenames[i] for i in incorrect_indices]
        incorrect_predictions = []
        for filename in incorrect_filenames:
            filepath = os.path.join(self.config.training_data, filename)
            true_label = true_classes[self.valid_generator.filenames.index(filename)]
            incorrect_predictions.append({
                "filepath": filepath,
                "label": true_label
            })

        return incorrect_predictions

    def save_incorrect_predictions(self, incorrect_predictions: list):
        self.logger.info("Saving incorrect predictions to incorrect_predictions.json...")
        save_json(path=Path("incorrect_predictions.json"), data={"data": incorrect_predictions})
        self.logger.info("Incorrect predictions saved successfully.")

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        self.logger.info("Saving evaluation scores to scores.json...")
        save_json(path=Path("scores.json"), data=scores)
        self.logger.info("Scores saved successfully.")
