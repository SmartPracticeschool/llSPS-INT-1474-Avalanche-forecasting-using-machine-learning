import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'test_heat':2, 'stickiness_score':1, 'snow':1})

print(r.json())