import requests
import numpy as np
import json

addr = 'http://localhost:5002'
test_url = addr + '/api/test'
content_type = 'application/json'
headers = {'content-type': content_type}

temp = np.zeros((5, 4)) + 0.1
temp = temp.tolist()
data = {'data': temp}

response = requests.post(
    test_url,
    json=json.dumps(data),
    headers=headers)
print(response.text)