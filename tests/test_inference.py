import requests
import json
import time

API_URL = 'http://127.0.0.1:5005/invocations'

sample_input_data = {
    'dataframe_split': {
        'columns': [
            'amount',
            'merchant_type_clothing',
            'merchant_type_electronics',
            'merchant_type_groceries',
            'merchant_type_others',
            'merchant_type_travel',
            'device_type_desktop',
            'device_type_mobile',
            'device_type_tablet'
        ],
        'data': [[
            0.6755338270184775,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0
        ]]
    }
}

headers = {'Content-Type': 'application/json'}
start = time.time()

try:
    res = requests.post(API_URL, headers=headers, json=sample_input_data)
    duration = time.time() - start
    if res.ok:
        print('Prediction:', res.json())
        print(f'Response time: {duration:.4f} sec')
    else:
        print(f'Error {res.status_code}:', res.text)
except Exception as e:
    print('Request failed:', str(e))