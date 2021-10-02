from key import API_KEY
import requests

response = requests.get(
    'http://api.exchangeratesapi.io/v1/2021-10-01?access_key={}&symbols=AUD,CAD,PLN,MXN'.format(API_KEY))
print(response.json())
