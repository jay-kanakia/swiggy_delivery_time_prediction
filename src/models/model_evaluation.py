import joblib
import logging
import dagshub
import mlflow
import json

import pandas as pd

from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,r2_score

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

def load_data(data_path:Path)->pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logging.error('File not found at the location')
    return df

def make_X_and_y(data:pd.DataFrame,target_column:str):

    X = data.drop(columns=[target_column])
    y = data[target_column]

    return X,y

def load_model(model_path:Path):
    
    model = joblib.load(model_path)

    return model

def save_model_info(save_json_path,run_id,artifact_path,model_name):
    info_dict = {
        'run_id' : run_id,
        'artifact_path' : artifact_path,
        'model_name' : model_name
    }

    with open(save_json_path,'w') as file:
        json.dump(info_dict,file,indent=4)

TARGET = 'time_taken'

if __name__ == "__main__":

    # root path
    root_path = Path(__file__).parent.parent.parent

    # data load path
    train_data_path = root_path/'data'/'processed'/'train_trans.csv'
    test_data_path = root_path/'data'/'processed'/'test_trans.csv'

    # model path
    #model_path = root_path/'models'/'preprocessor.joblib'

    # load the training and test data
    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)
    logging.info('Train and test data loaded successfully')

    # split the train and test data
    X_train,y_train = make_X_and_y(data=train_data,target_column=TARGET)
    X_test,y_test = make_X_and_y(data=test_data,target_column=TARGET)

    