import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from src.utils.utils import save_json
from pathlib import Path
import joblib
from src.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average="weighted")
        recall = recall_score(actual, pred, average="weighted")
        f1 = f1_score(actual, pred, average="weighted")
        return accuracy, precision, recall, f1

    def save_results(self):
        # Load test data and trained model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        # Separate features and target variable
        test_x = test_data.drop(columns=[self.config.target_column])
        test_y = test_data[self.config.target_column]  # ✅ Classification: Keep as Series

        # Make predictions
        predicted_labels = model.predict(test_x)

        # Compute classification metrics
        accuracy, precision, recall, f1 = self.eval_metrics(test_y, predicted_labels)

        # Save metrics as JSON
        scores = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        save_json(path=Path(self.config.metric_file_name), data=scores)

        print("Model evaluation completed! Metrics saved successfully.")
        print("Classification Report:\n", classification_report(test_y, predicted_labels))
