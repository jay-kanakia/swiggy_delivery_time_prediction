import pandas as pd
import yaml
import logging

from sklearn.model_selection import train_test_split
from pathlib import Path

# create logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.INFO)

# create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# add handler to logger
logger.addHandler(console_handler)

def load_data(data_path : Path)->pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    
    except FileNotFoundError:
        logger.error('The file not found at the given location')

    return df

def read_params(file_path : Path):

    with open(file_path,'r') as file:
        params_file = yaml.safe_load(file)

    return params_file

def split_data(data : pd.DataFrame,test_size : float, random_state : int)->pd.DataFrame:

    train_data, test_data = train_test_split(data,test_size=test_size,random_state=random_state)

    return train_data,test_data

def save_data(data:pd.DataFrame,saved_data_path : Path)->None:

    data.to_csv(saved_data_path,index=False)

if __name__ == '__main__':

    # root path
    root_path = Path(__file__).parent.parent.parent

    # data load path
    data_path = root_path/'data'/'interim'/'swiggy_cleaned.csv'

    #save data directory
    save_data_dir = root_path/'data'/'external'

    # make dir if not exist
    save_data_dir.mkdir(exist_ok=True,parents=True)

    # train filename
    train_filename = 'train.csv'

    # test filename
    test_filename = 'test.csv'

    # save path
    save_train_path = save_data_dir/train_filename
    save_test_path = save_data_dir/test_filename

    # parameters file
    param_file = root_path/'params.yaml'

    #load the clean data
    df = load_data(data_path)
    logger.info('Data loaded successfully')

    # read the parameters
    parameters = read_params(param_file)['data_preprocessing']
    test_size = parameters['test_size']
    random_state = parameters['random_state']   
    logger.info('Parameters loaded successfully')

    # split train test
    train_data,test_data = split_data(df,test_size,random_state)
    logger.info('Data splitted to train and test')

    # save the train and test dat
    data_subset = [train_data,test_data]
    data_paths = [save_train_path,save_test_path]
    filename_list = [train_filename,test_filename]

    for filename, path, data in zip(filename_list,data_paths,data_subset):
        save_data(data=data,saved_data_path=path)
        logger.info(f"{filename.replace('.csv','')} data saved to location")





