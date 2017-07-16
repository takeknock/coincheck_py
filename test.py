import requests
import yaml
from bs4 import BeautifulSoup

f = open("init.yml", "r+")
login_data = yaml.load(f)


payload = {
    'utf8' : '✓',
    'type' : 'consumer',
    'email' : login_data['email'],
    'password' : login_data['password']
}
# need to add authentication
print(login_data)



s = requests.session()
r = s.get("https://coincheck.com/sessions/signin")
soup = BeautifulSoup(r.text)
auth_token = soup.find(attrs={'name': 'authenticity_token'}).get('value')
payload['authenticity_token'] = auth_token

s.post("https://coincheck.com/sessions/signin")

