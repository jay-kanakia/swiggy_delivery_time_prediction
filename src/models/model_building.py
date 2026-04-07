import pandas as pd

import joblib
import yaml
import logging

from pathlib import Path
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer

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
        logger.error('File not found at the location')

    return df

def read_params(param_path:Path):

    with open(param_path,'r') as file:
        params_file = yaml.safe_load(file)

    return params_file

def save_model(model,save_model_dir:Path,model_name:str):

    save_location = save_model_dir/model_name
    joblib.dump(value=model,filename=save_location)

def save_transformer(transformer,save_transformer_dir:Path,transformer_name:str):

    save_location = save_transformer_dir/transformer_name

    joblib.dump(value=transformer,filename=save_location)

def train_model(model,X_train:pd.DataFrame,y_train):

    model.fit(X_train,y_train)

    return model

def make_X_and_y(data:pd.DataFrame,target_column:str):

    X = data.drop(columns=target_column)
    y = data[target_column]

    return X,y

TARGET = 'time_taken'

if __name__ == '__main__':

    # root path
    root_path = Path(__file__).parent.parent.parent

    # load train data
    train_data_path = root_path/'data'/'processed'/'train_trans.csv'

    # params path
    params_file_path = root_path/'params.yaml'

    # loading training data
    train_data = load_data(train_data_path)
    logger.info('Train data loaded successfully')

    # split the data to X and y
    X_train,y_train = make_X_and_y(train_data,target_column=TARGET)
    logger.info('Dataset splitting completed')

    # models parameters
    model_params = read_params(param_path=params_file_path)['train']
    rf_params = model_params['random_forest']
    logger.info('Random Forest parameters read')

    # build random forest model
    rfg = RandomForestRegressor(**rf_params)
    logger.info('Random forest build')

    lgbm_params = model_params['light_gbm']
    lgbm = LGBMRegressor(**lgbm_params)
    logger.info('LightGBM model build')

    # meta model
    lr = LinearRegression()
    logger.info('LR(meta) model build')

    # power transformer
    pt = PowerTransformer(method='yeo-johnson')
    logger.info('Target transfomer build')

    # form stacking regressor
    stacking_regressor = StackingRegressor(
        estimators=[('rf_model',rfg),
                    ('light_gbm',lgbm)],
                    final_estimator=lr,
                    cv=5,n_jobs = -1
    )
    logger.info('Stacking regresor built')

    # make model wrapper
    model = TransformedTargetRegressor(regressor=stacking_regressor,transformer=pt)
    logger.info('Model wrapped inside a wrapper')

    # fit the model on training data
    train_model(model,X_train,y_train)
    logger.info('Model training completed')

    # extract the model from wrapper
    stacking_model = model.regressor_
    transformer = model.transformer_

    # save model
    model_save_dir = root_path/'models'
    model_save_dir.mkdir(exist_ok=True,parents=True)
    model_filename = 'model.joblib'

    save_model(model=model,save_model_dir=model_save_dir,model_name=model_filename)
    logger.info('Trained model save to location')

    # save stacking model
    stacking_filename = "stacking_regressor.joblib"
    save_model(model=stacking_model,save_model_dir=model_save_dir,model_name=stacking_filename)
    logger.info("Trained model saved to location")
    
    # save the transformer
    transformer_filename = "power_transformer.joblib"
    save_transformer(transformer, model_save_dir, transformer_filename)
    logger.info("Transformer saved to location")

    


    