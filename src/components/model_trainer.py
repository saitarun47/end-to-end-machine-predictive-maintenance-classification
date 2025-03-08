
import pandas as pd
import os
from src import logger
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.entity.config_entity import ModelTrainerConfig



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):
        # Load train & test data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Separate features and target
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]  # Use Series, not DataFrame
        test_y = test_data[self.config.target_column]

        # Initialize RandomForestClassifier
        rf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            criterion=self.config.criterion,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
            class_weight=self.config.class_weight
        )

        # Train the model
        rf.fit(train_x, train_y)

        # Save trained model
        joblib.dump(rf, os.path.join(self.config.root_dir, self.config.model_name))

        print("Random Forest Model Trained and Saved Successfully!")
