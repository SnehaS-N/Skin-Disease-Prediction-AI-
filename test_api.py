import requests

url = "http://127.0.0.1:5000/predict"

with open("test.jpg", "rb") as img:
    files = {"image": img}
    response = requests.post(url, files=files)

print(response.status_code)
print(response.json())