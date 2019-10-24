import requests
import numpy as np
import json

'''
show drum roll in console
'''
def printDrumRoll(roll):
    trans = np.flip(np.transpose(roll), 0)
    for r_i, r in enumerate(trans):
        print('[{}]'.format(8 - r_i), end='')
        for i, w in enumerate(r):
            if i > 0 and i % 24 == 0:
                print('|', end='')
            if w < 0.2:
                print('_', end='')
            else:
                print('*', end='')
        print()

def printLatent(lt):
    print('latent:')
    for i, value in enumerate(lt):
        print('[{}]: {}'.format(i, value))
        
addr = 'http://localhost:5002'
test_url = addr + '/static'
content_type = 'application/json'
headers = {'content-type': content_type}

response = requests.get(
    test_url,
    headers=headers)

r_json = json.loads(response.text)
drum_rolls = r_json['result']
latent = r_json['latent']

printDrumRoll(drum_rolls)
printLatent(latent)

