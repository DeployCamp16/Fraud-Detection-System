import requests
import json
import time
from joblib import load 
import pandas as pd 

API_URL = 'http://127.0.0.1:5005/invocations'
processor = load("../models/preprocessing/preprocessor.joblib")  
column_list = pd.read_csv("../data/processed/column_list_processed.csv").columns.tolist()

sample_json_data = {
    "transaction_id" : 1,
    "amount" : 39.35,
    "merchant_type" : "clothing",
    "device_type" : "desktop"
}

headers = {'Content-Type': 'application/json'}
start = time.time()

try:
    input_dataframe = pd.DataFrame([sample_json_data])
    transformed_data = processor.transform(input_dataframe).tolist()

    input_data =  {
        'dataframe_split': {
            'columns': column_list,
            'data': transformed_data
        }
    }

    res = requests.post(API_URL, headers=headers, json=input_data)
    duration = time.time() - start
    if res.ok:
        print('Prediction:', res.json())
        print(f'Response time: {duration:.4f} sec')
    else:
        print(f'Error {res.status_code}:', res.text)
except Exception as e:
    print('Request failed:', str(e))