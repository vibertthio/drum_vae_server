import os
import time
from flask import Flask, request, Response
from flask_cors import CORS
import numpy as np
import json
import pypianoroll
from pypianoroll import Multitrack, Track
import torch.utils.data as Data
from vae_rnn import *

app = Flask(__name__)
app.config['ENV'] = 'development'
CORS(app)
dims = [3, 2]


'''
constants
'''
STEP_UNIT = 0.0015
GLOBAL_THRESHOLD = 0.2


'''
laod model
'''
path = '/home/vibertthio/local_dir/vibertthio/drum_generation/server/models/'
model = [m for m in os.listdir(path) if '.pt' in m][0]
encoder = Encoder().to(device)
decoder = Decoder().to(device)
vae = VAE(encoder, decoder).to(device)
vae.load_state_dict(torch.load(path + model))


'''
load data
'''
genres = [x for x in os.listdir(
    '/home/vibertthio/local_dir/vibertthio/drum_generation/server/data/') if '.npy' in x]
train_x_np = np.load(
    '/home/vibertthio/local_dir/vibertthio/drum_generation/server/data/' + genres[7])
train_x = torch.from_numpy(train_x_np).type(torch.FloatTensor)
train_dataset = Data.TensorDataset(train_x)
train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=1,
)


'''
load preset
'''
fn_latent_selected = './static/static_20181015_235336.npy'
latent_selected_np = np.load(fn_latent_selected)


'''
utils
'''
def printDrumRoll(roll):
    trans = np.flip(np.transpose(roll), 0)
    for r_i, r in enumerate(trans):
        print('[{}]'.format(8 - r_i), end='')
        for i, w in enumerate(r):
            if i > 0 and i % 16 == 0:
                print('|', end='')
            if w == 0:
                print('_', end='')
            else:
                print('*', end='')
        print()

def savaTensor2Array(tensor):
    arr = tensor.cpu().data.numpy()
    t = time.strftime("%Y%m%d_%H%M%S")
    filename = './static/static_' + t + '.npy'
    np.save(filename, arr)

def decodeLatent(latent, send_latent=True):
    '''
    Return the jsonified object of two things:
    1) the drum arrays around the specified latent
    2) the latent vector

    Keyword arguments:
    latent -- the latent vector of the center point
    '''
    out = decoder(latent)
    out_np = out.cpu().data.numpy()
    out_np = np.where(out_np > GLOBAL_THRESHOLD, 128, 0)
    out_concat = np.zeros((9, 96, 9))
    out_latent = []
    
    for i in range(9):
        x = i % 3 - 1
        y = i // 3 - 1

        latent_shift = latent.cpu().data.numpy()
        shift = np.zeros(latent_shift.shape, dtype=np.float32)
        shift[0][dims[1]] = x * STEP_UNIT * 100
        shift[0][dims[0]] = y * STEP_UNIT * 100
        latent_shift = latent_shift + shift

        latent_shift = torch.from_numpy(latent_shift).to(device)
        o = decoder(latent_shift)
        out_latent.append(latent_shift.cpu().data.numpy()[0].tolist())

        o = o[0].cpu().data.numpy()
        o = np.where(o > 0.5, 1, 0)
        out_concat[i] = o

    out_concat = out_concat.tolist()
    response = {
        'result': out_concat,
        'latent': out_latent,
    }
    response_pickled = json.dumps(response)
    return response_pickled


'''
api route
'''
@app.route('/rand', methods=['POST', 'GET'])
def rand():
    '''random
    1. get next sample from training data
    2. return the outputs around the random sample
    '''
    with torch.no_grad():
        global latent_selected_np
        data = iter(train_loader).next()[0]

        data = Variable(data).type(torch.float32).to(device)
        latent = vae._enc_mu(encoder(data))
        latent_selected_np = latent.cpu().data.numpy()

        # save data
        # savaTensor2Array(latent)

    response_pickled = decodeLatent(latent)
    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/static', methods=['GET'], endpoint='static_1')
def static():
    with torch.no_grad():
        global latent_selected_np
        global dims
        latent_selected_np = np.load(fn_latent_selected)
        dims = [3, 2]
        latent = torch.from_numpy(latent_selected_np).to(device)

    response_pickled = decodeLatent(latent)
    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/dim/<d1>/<d2>', methods=['GET'], endpoint='dim_1')
def change_dim(d1, d2):
    with torch.no_grad():
        ''' change the choosen dimension '''
        global latent_selected_np
        global dims
        dims[0] = int(d1)
        dims[1] = int(d2)
        latent = torch.from_numpy(latent_selected_np).to(device)

    response_pickled = decodeLatent(latent)
    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/static/<dir>', methods=['GET'], endpoint='static_direction_1', defaults={'step': str(STEP_UNIT)})
@app.route('/static/<dir>/<step>', methods=['GET'], endpoint='static_direction_1')
def static_direction(dir, step):
    '''Move the selected latent in the dims choosen

    Keyword arguments:
    dir -- the direction of moving
    step -- the distance of each step
    '''
    with torch.no_grad():
        global latent_selected_np
        stp = float(step)
        if dir == '0':
            latent_selected_np[0][dims[0]] += stp
        elif dir == '1':
            latent_selected_np[0][dims[0]] -= stp
        elif dir == '2':
            latent_selected_np[0][dims[1]] += stp
        elif dir == '3':
            latent_selected_np[0][dims[1]] -= stp
        latent = torch.from_numpy(latent_selected_np).to(device)

    response_pickled = decodeLatent(latent)
    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/adjust/<dim>/<value>', methods=['GET'], endpoint='adjust_latent_1')
def adjust_latent(dim, value):
    '''Adjust the selected dimension of the latent 

    Keyword arguments:
    dim -- the dimension adjusted
    value -- the value assigned
    '''
    with torch.no_grad():
        global latent_selected_np
        d = int(float(dim))
        v = float(value)
        latent_selected_np[0][d] = v
        latent = torch.from_numpy(latent_selected_np).to(device)

    response_pickled = decodeLatent(latent)
    return Response(response=response_pickled, status=200, mimetype="application/json")


'''
start app
'''
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
