import requests

url = "http://127.0.0.1:8000/predict"
files = {'file': open('/home/joelbrice/code/joelbrice/brainmap/data/raw_data/Testing/glioma/Te-gl_0010.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
