import pandas as pd
import requests

# fetching one sample row from the data
url = 'https://raw.githubusercontent.com/Himanshu-1703/swiggy-delivery-time-prediction/refs/heads/main/swiggy%20dataset/swiggy.csv'

df = pd.read_csv(url).dropna()

sample_row = df.sample(1)
print("The target value is",sample_row.iloc[:,-1].values.item().replace("(min) ",""))

# remove the target column
data = sample_row.drop(columns=sample_row.columns.tolist()[-1]).squeeze().to_dict()
#print(data)

# predict url
predict_url = "http://127.0.0.1:8000/predict"

# get the respone from api
response = requests.post(url=predict_url,json=data)

# test for 200 code
print("The status code for response is", response.status_code)

if response.status_code == 200:
    print(f"The prediction value by the API is {float(response.text):.2f} min")
else:
    print("Error:", response.status_code, f"{response.text}")