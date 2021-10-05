import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE

if __name__ == "__main__":

    if not os.path.exists('data/word_embeddings.npy'):
        raise FileNotFoundError("word embeddings file was't found")

    parser = argparse.ArgumentParser(description='RVAE')
    parser.add_argument('--num-iterations', type=int, default=120000, metavar='NI',
                        help='num iterations (default: 120000)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('--start-epoch', type=int, default=50000, metavar='EP',
                        help='dropout (default: 50000)')
    parser.add_argument('--train-data', default='', metavar='TD',
                        help='load custom training dataset (default: '')')
    # parser.add_argument('--ce-result', default='', metavar='CE',
    #                     help='ce result path (default: '')')
    # parser.add_argument('--kld-result', default='', metavar='KLD',
    #                     help='ce result path (default: '')')


    args = parser.parse_args()

    batch_loader = BatchLoader(path = '', 
                               custom_index = False, 
                               train_data_name=args.train_data)

    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    rvae = RVAE(parameters)
    optimizer = Adam(rvae.learnable_parameters(), args.learning_rate)

    if args.use_trained:
        rvae.load_state_dict(
            t.load('./trained_model/{}_trained_{}'.format(
                args.train_data.split('.')[0], args.start_epoch)))
        optimizer.load_state_dict(
            t.load('./trained_model/{}_trained_optimizer_{}'.format(
                args.train_data.split('.')[0], args.start_epoch)))

    if args.use_cuda:
        rvae = rvae.cuda()

    train_step = rvae.trainer(optimizer, batch_loader)
    validate = rvae.validater(batch_loader)

    ce_result = []
    kld_result = []

    for iteration in range(args.start_epoch, args.num_iterations):

        cross_entropy, kld, coef = train_step(iteration, args.batch_size, args.use_cuda, args.dropout)

        if iteration % 5 == 0:
            print('\n')
            print('------------TRAIN-------------')
            print('----------ITERATION-----------')
            print(iteration)
            print('--------CROSS-ENTROPY---------')
            print(cross_entropy.data.cpu().numpy())
            print('-------------KLD--------------')
            print(kld.data.cpu().numpy())
            print('-----------KLD-coef-----------')
            print(coef)
            print('------------------------------')

        if iteration % 10 == 0:
            cross_entropy, kld = validate(args.batch_size, args.use_cuda)

            cross_entropy = cross_entropy.data.cpu().numpy()
            kld = kld.data.cpu().numpy()

            print('\n')
            print('------------VALID-------------')
            print('--------CROSS-ENTROPY---------')
            print(cross_entropy)
            print('-------------KLD--------------')
            print(kld)
            print('------------------------------')

            ce_result += [cross_entropy]
            kld_result += [kld]

        if iteration % 20 == 0:
            seed = np.random.normal(size=[1, parameters.latent_variable_size])

            sample = rvae.sample(batch_loader, 50, seed, args.use_cuda)

            print('\n')
            print('------------SAMPLE------------')
            print('------------------------------')
            print(sample)
            print('------------------------------')

        if iteration % 25000 == 0:
            t.save(rvae.state_dict(), './trained_model/{}_trained_{}'.format(
                args.train_data, iteration))
            t.save(optimizer.state_dict(), './trained_model/{}_trained_optimizer_{}'.format(
                args.train_data, iteration))

    # np.save('ce_result_{}.npy'.format(args.ce_result), np.array(ce_result))
    # np.save('kld_result_npy_{}'.format(args.kld_result), np.array(kld_result))
