import requests

url = 'https://api.spacexdata.com/v3/capsules/C112'
payload = {}
headers = {}
response = requests.request('GET', url, headers = headers, data = payload, allow_redirects=False)
print(response.text)
