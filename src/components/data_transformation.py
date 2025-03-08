import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src import logger
from src.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.scaler = StandardScaler()  # Initialize StandardScaler
        self.encoders = {}  # Dictionary to store label encoders for categorical columns

    def encode_categorical(self, df, categorical_columns):
        """Encodes categorical columns using Label Encoding."""
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])  # Encode categorical column
                self.encoders[col] = le  # Store encoder for inverse transformation if needed
        return df

    def scale_features(self, df, numerical_columns):
        """Scales numerical features using StandardScaler."""
        df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        return df

    def train_test_splitting(self):
        """Reads data, drops ID columns, applies encoding & scaling, then splits into train-test sets."""
        # Load data
        data = pd.read_csv(self.config.data_path)

        # Drop ID columns
        data.drop(columns=["UDI", "Product ID","Failure Type"], axis=1, inplace=True)

        # Identify categorical and numerical columns
        categorical_columns = ["Type", "Failure Type"]
        numerical_columns = [
            "Air temperature [K]", 
            "Process temperature [K]", 
            "Rotational speed [rpm]", 
            "Torque [Nm]", 
            "Tool wear [min]"
        ]

        # Encode categorical features
        data = self.encode_categorical(data, categorical_columns)

        # Scale numerical features
        data = self.scale_features(data, numerical_columns)

        # Train-test split
        train, test = train_test_split(data, test_size=0.25, random_state=42)

        # Save processed files
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Data processed: ID columns dropped, encoding & scaling applied, split into train-test sets")
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}")

        print(f"Train Shape: {train.shape}")
        print(f"Test Shape: {test.shape}")
