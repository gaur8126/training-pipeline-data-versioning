import os 
import pandas as pd
import numpy as np 
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml 

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# Logging Configuration

logger = logging.getLogger('model_training')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "model_training.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file"""

    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        logger.debug("Parameters retrieved from %s", params_path)
        return params
    
    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpecteed error: %s", e)
        raise 



def load_data(file_path:str) -> pd.DataFrame:
    """
    Load the data from a CSV file

    :param file_path: Path to the CSV file
    :return: Loaded Dataframe
    """

    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s",file_path, df.shape)
        return df 
    
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s",e)
        raise
    except FileNotFoundError  as e:
        logger.error("File not found: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading the data: %s",e)
        raise

def train_model(x_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:

    """
    Train the RandomForest Model.

    :param x_train: Training features
    :param y_train: Training Label
    "param params: Disctionary of hyperparameters
    :return: Trained RandomForestClassifier
    """

    try:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in x_train and y_train must be the same")
        
        logger.debug("Initializing RandomForest model with parameters: %s",params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])

        logger.debug("Model training started with %d samples",x_train.shape[0])
        clf.fit(x_train, y_train)
        logger.debug("Model traning completed")

        return clf
    except ValueError as e:
        logger.error("ValueError during model training: %s",e)
        raise
    except Exception as e:
        logger.error("Error during model training")
        raise 


def save_mdoel(model, file_path:str) -> None:

    """
    save the trained model to a file,
    :param model: Trained model object
    :param file_path: Path to save the model file
    """

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok= True)

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error("File path not found: %s",e)
        raise
    except Exception as e:
        logger.error("Error occured while saving the model: %s",e)
        raise 

def main():

    try:
        params = load_params('params.yaml')['model_training']
        # params = {'n_estimators':25, 'random_state':2}
        train_data = load_data('./data/processed/train_tfidf.csv')
        x_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values

        clf = train_model(x_train, y_train, params)

        model_save_path = 'model/model.pkl'
        save_mdoel(clf, model_save_path)

    except Exception as e:
        logger.error("Failed to complete the model building process: %s",e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
