import argparse
import os
import pickle

import numpy as np
import torch as t

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sampler')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--num-sample', type=int, default=10, metavar='NS',
                        help='num samplings (default: 10)')
    parser.add_argument('--model-name', default='', metavar='TD',
                        help='name of saved model (default: '')')
    parser.add_argument('--train-data', default='', metavar='TD',
                        help='load custom training dataset (default: '')')


    args = parser.parse_args()

    batch_loader = BatchLoader(path = '', 
                               custom_index = False, 
                               train_data_name=args.train_data)

    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    rvae = RVAE(parameters)
    rvae.load_state_dict(t.load('./trained_model/{}'.format(args.model_name)))
    if args.use_cuda:
        rvae = rvae.cuda()

    sents = []
    seeds = {}

    for iteration in range(args.num_sample):
        seed = np.random.normal(size=[1, parameters.latent_variable_size])
        sent = rvae.sample(batch_loader, 50, seed, args.use_cuda)
        print(sent)
        sents.append(sent)
        seeds[sent] = seed.flatten()

    with open('./generated/rnd_{}_{}_{}.txt'.format(args.model_name, 
                                                args.train_data, 
                                                args.num_sample),'w') as f:
        f.writelines('\n'.join(sents))

    pickle.dump(seeds, open('./generated/rnd_{}_{}_{}_zdict.pkl'.format(args.model_name, 
                                                args.train_data, 
                                                args.num_sample),'wb'))