import requests, json

url = "bugs.python.org"

payload = {"number": 12524, 
                   "type": "issue", 
                              "action": "show"}

header = {"Content-type": "application/x-www-form-urlencoded",
                  "Accept": "text/plain"} 

response_decoded_json = requests.post(url, data=payload, headers=header)
response_json = response_decoded_json.json()

print response_json
