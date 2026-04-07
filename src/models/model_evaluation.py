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

# dagshub initialization
dagshub.init(repo_owner='jay-kanakia', repo_name='swiggy_delivery_time_prediction', mlflow=True)
# set tracking uri
mlflow.set_tracking_uri('https://dagshub.com/jay-kanakia/swiggy_delivery_time_prediction.mlflow')
# set experiment
mlflow.set_experiment(experiment_name='DVC pipeline Exp')

# setting target
TARGET = 'time_taken'

if __name__ == "__main__":

    # root path
    root_path = Path(__file__).parent.parent.parent

    # data load path
    train_data_path = root_path/'data'/'processed'/'train_trans.csv'
    test_data_path = root_path/'data'/'processed'/'test_trans.csv'

    # load the training and test data
    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)
    logging.info('Train and test data loaded successfully')

    # split the train and test data
    X_train,y_train = make_X_and_y(data=train_data,target_column=TARGET)
    X_test,y_test = make_X_and_y(data=test_data,target_column=TARGET)

    # loading model
    model_path = root_path/'models'/'model.joblib'
    model = load_model(model_path)
    logger.info('Model loaded successfully')

    # get the prediction
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    logger.info('Prediction in model completed')

    # MAE calculation
    train_mae = mean_absolute_error(y_train,y_pred_train)
    test_mae = mean_absolute_error(y_test,y_pred_test)
    logger.info('Error calculated')

    # r2 score
    train_r2 = r2_score(y_train,y_pred_train)
    test_r2 = r2_score(y_test,y_pred_test)
    logger.info('r2 score calculated')

    # calculate cross val score
    score = cross_val_score(model,X_train,y_train,cv=5,scoring='neg_mean_absolute_error',n_jobs=-1)
    logger.info('cross val completed')

    # mean cross val score
    mean_score  = score.mean()

    # logging with mlflow

    with mlflow.start_run(run_name='dvc_model') as run:

        # set tag
        mlflow.set_tag('model','Food delivery time regressor')

        # log params
        mlflow.log_params(model.get_params())

        # log metrics
        mlflow.log_metric("train_mae",train_mae)
        mlflow.log_metric("test_mae",test_mae)
        mlflow.log_metric("train_r2",train_r2)
        mlflow.log_metric("test_r2",test_r2)
        mlflow.log_metric("mean_cv_score",-(score.mean()))

        # log individual scores
        mlflow.log_metrics(
            {
                f"CV {num}" : score for num,score in enumerate(-score)
            }
        )

        # mlflow dataset input datatype
        train_data_input = mlflow.data.from_pandas(train_data,targets=TARGET)
        test_data_input = mlflow.data.from_pandas(test_data,targets=TARGET)

        # log input
        mlflow.log_input(dataset=train_data_input,context='training')
        mlflow.log_input(dataset=test_data_input,context='validation')

        # model signature
        model_signature = mlflow.models.infer_signature(model_input=X_train.sample(20,random_state=42),
                                                        model_output=model.predict(X_train.sample(20,random_state=42)))
        
        #log the final model
        mlflow.sklearn.log_model(model,'delivery_time_pred_model',signature=model_signature)


        # log stacking regressor
        mlflow.log_artifact(root_path / "models" / "stacking_regressor.joblib")
        
        # log the power transformer
        mlflow.log_artifact(root_path / "models" / "power_transformer.joblib")
        
        # log the preprocessor
        mlflow.log_artifact(root_path / "models" / "preprocessor.joblib")

        # get the current run artifact uri
        artifact_uri = mlflow.get_artifact_uri()

        logger.info("Mlflow logging complete and model logged")

    # get the run info
    run_id = run.info.run_id
    model_name = 'delivery_time_pred_model'

    # save the model info
    save_json_path = root_path / "run_information.json"
    save_model_info(save_json_path=save_json_path,
                    run_id=run_id,
                    artifact_path=artifact_uri,
                    model_name=model_name)
    logger.info("Model Information saved")





    