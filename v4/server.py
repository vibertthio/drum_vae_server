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

'''
app setup
'''
app = Flask(__name__)
app.config['ENV'] = 'development'
CORS(app)

'''
constants
'''
STEP_UNIT = 0.0015
GLOBAL_THRESHOLD = 0.2


'''
laod model
'''
path = '/home/vibertthio/local_dir/vibertthio/drum_generation/server/v4/models/conditional/'
model = [m for m in os.listdir(path) if '.pt' in m][0]
# model = 'vae_L1E-02_beta2E+01_beat48_loss2E+01_tanh_gru32_e10_b256_hd64-32_20181210_152332.pt'
encoder = Encoder().to(device)
decoder = Decoder().to(device)
vae = VAE(encoder, decoder).to(device)
vae.load_state_dict(torch.load(path + model))


'''
load data
'''
genre = 10
genres = [x for x in os.listdir(
    '/home/vibertthio/local_dir/vibertthio/drum_generation/server/data/') if '.npy' in x]
print(genres)
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
data_np = decoder(torch.from_numpy(
    latent_selected_np).to(device), genre).cpu().data.numpy()
data_np_orig = np.copy(data_np)

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
            if w < 0.2:
                print('_', end='')
            else:
                print('*', end='')
        print()

def savaTensor2Array(tensor):
    arr = tensor.cpu().data.numpy()
    t = time.strftime("%Y%m%d_%H%M%S")
    filename = './static/static_' + t + '.npy'
    np.save(filename, arr)

def decodeLatent(latent, update_data=False):
    '''
    Return the jsonified object of two things:
    1) the drum arrays of the latent
    2) the latent vector

    Keyword arguments:
    latent -- the latent vector of the center point
    '''
    out = decoder(latent, genre).cpu().data.numpy()
    out_result = out[0].tolist()
    out_latent = latent.cpu().data.numpy()[0].tolist()

    if update_data:
        global data_np
        data_np = out

    response = {
        'result': out_result,
        'latent': out_latent,
    }
    response_pickled = json.dumps(response)
    return response_pickled

def encodeData(data, update_latent=True):
    '''
    Return the jsonified object of two things:
    1) the drum arrays of the latent
    2) the latent vector

    Keyword arguments:
    latent -- the latent vector of the center point
    '''
    latent = vae._enc_mu(encoder(data)).cpu().data.numpy()
    out_result = data.cpu().data.numpy()[0].tolist()
    out_latent = latent[0].tolist()

    if update_latent:
        global latent_selected_np
        latent_selected_np = latent

    response = {
        'result': out_result,
        'latent': out_latent,
    }
    response_pickled = json.dumps(response)
    return response_pickled


'''
api route
'''
@app.route('/rand', methods=['GET'])
def rand():
    '''random
    1. get next sample from training data
    2. return the outputs around the random sample
    '''
    with torch.no_grad():
        global latent_selected_np
        global data_np
        data = iter(train_loader).next()[0]
        data_np = data.cpu().data.numpy()

        data = Variable(data).type(torch.float32).to(device)
        latent = vae._enc_mu(encoder(data))
        latent_selected_np = latent.cpu().data.numpy()

        response_pickled = decodeLatent(latent)
        return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/static', methods=['GET'], endpoint='static_1')
def static():
    with torch.no_grad():
        global latent_selected_np
        global data_np
        data_np = np.copy(data_np_orig)
        latent_selected_np = np.load(fn_latent_selected)
        latent = torch.from_numpy(latent_selected_np).to(device)

        response_pickled = decodeLatent(latent)
        return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/adjust-latent/<dim>/<value>', methods=['GET'], endpoint='adjust_latent_1')
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

        response_pickled = decodeLatent(latent, update_data=True)
        return Response(response=response_pickled, status=200, mimetype="application/json")


@app.route('/adjust-data/<i>/<j>/<value>', methods=['GET'], endpoint='adjust_data_1')
def adjust_data(i, j, value):
    '''Adjust the selected dimension of the latent 

    Keyword arguments:
    i, j -- the adjusted position of data
    value -- the value assigned
    '''
    with torch.no_grad():
        global latent_selected_np
        global data_np

        i_int = int(float(i))
        j_int = int(float(j))
        value_float = float(value)

        data_np[0][i_int][j_int] = value_float
        data_np = np.float32(np.where(data_np > 0.2, 1.0, 0))
        data = torch.from_numpy(data_np).to(device)

        response_pickled = encodeData(data)
        return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/adjust-genre/<g>', methods=['GET'], endpoint='adjust_genre_1')
def adjust_latent(g):
    '''Adjust the selected dimension of the latent

    Keyword arguments:
    dim -- the dimension adjusted
    value -- the value assigned
    '''
    with torch.no_grad():
        global genre
        genre = int(float(g))
        latent = torch.from_numpy(latent_selected_np).to(device)

        response_pickled = decodeLatent(latent, update_data=True)
        return Response(response=response_pickled, status=200, mimetype="application/json")



'''
start app
'''
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5010)
