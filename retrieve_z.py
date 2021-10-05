import argparse
import os
import pickle

import numpy as np
import torch as t

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE

if __name__ == '__main__':

    #assert os.path.exists('trained_RVAE'), \
    #    'trained model not found'

    parser = argparse.ArgumentParser(description='Sampler')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--batch-size', type=int, default=10, metavar='NS',
                        help='num samplings (default: 10)')
    parser.add_argument('--sample-data', default='', metavar='TD',
                        help='load custom training dataset (default: '')')
    parser.add_argument('--model-name', default='', metavar='TD',
                        help='name of saved model (default: '')')

    args = parser.parse_args()

    batch_loader = BatchLoader('' , custom_index=True, train_data_name=args.sample_data)
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    rvae = RVAE(parameters)
    rvae.load_state_dict(t.load('./trained_model/{}'.format(args.model_name)))
    if args.use_cuda:
        rvae = rvae.cuda()

    sampler = rvae.latent_sampler(batch_loader)

    zs = {}
    for i in range(0, int(batch_loader.total_lines('train')/args.batch_size)+1):
        indexes = np.array(range(i*args.batch_size,min((i+1)*args.batch_size, batch_loader.total_lines('train'))))
        if len(indexes) > 0:
            z = sampler(indexes, args.use_cuda)
            z = z.cpu().data.numpy()
            print(z, z.shape) 
           # zs.append([(indexes[i], z[i]) for i in range(len(indexes))])
            for i in range(len(indexes)):
                zs[indexes[i]] = z[i]
            print('sampling from index: {} => total: {}'.format(indexes, len(zs)))

    # z = np.concatenate(zs, axis = 0)
    pickle.dump(zs, open('./cscw_data/{}_z.pkl'.format(args.sample_data),'wb'))
