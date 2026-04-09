import pandas as pd
import requests
import pytest

# fetching one sample row from the data
url = 'https://raw.githubusercontent.com/Himanshu-1703/swiggy-delivery-time-prediction/refs/heads/main/swiggy%20dataset/swiggy.csv'

df = pd.read_csv(url).dropna()

sample_row = df.sample(1)
print("The target value is",sample_row.iloc[:,-1].values.item().replace("(min) ",""))

# remove the target column
data = sample_row.drop(columns=sample_row.columns.tolist()[-1]).squeeze().to_dict()

@pytest.mark.parametrize(argnames='url,data',argvalues=[("http://127.0.0.1:8000/predict", data)])
def test_predict_endpoint(url,data):

    # get the respone from api
    response = requests.post(url=url,json=data)

    # test for 200 code
    assert response.status_code == 200, "Prediction endpoint not giving response"