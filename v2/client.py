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
            if w == 0:
                print('_', end='')
            else:
                print('*', end='')
        print()

def printLatent(lt):
    print('latent:')
    for i, value in enumerate(lt):
        print('[{}]: {}'.format(i, value))
        
addr = 'http://localhost:5001'
test_url = addr + '/rand'
content_type = 'application/json'
headers = {'content-type': content_type}

response = requests.get(
    test_url,
    headers=headers)

r_json = json.loads(response.text)
drum_rolls = r_json['result']
latent = r_json['latent']

for i, d in enumerate(drum_rolls):
    x = i % 3 - 1
    y = i // 3 - 1
    if x == 0 and y == 0:
        print('({}, {})'.format(x, y))
        printDrumRoll(d)
        printLatent(latent[i])
