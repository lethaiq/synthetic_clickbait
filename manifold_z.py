import pickle
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import NMF, LatentDirichletAllocation

import umap


plt.get_current_fig_manager().window.showMaximized()


if __name__ == '__main__':
	sample_data = 'P_X_train_pos_text'

	parser = argparse.ArgumentParser(description='Sampler')
	parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
	                    help='use cuda (default: True)')
	parser.add_argument('--alg', default='pca', metavar='ALG',
	                    help='load custom training dataset (default: "PCA")')

	args = parser.parse_args()

	z = pickle.load(open('./cscw_data/{}_z.pkl'.format(sample_data),'rb'))
	z = np.array([z[i] for i in z])
	
	P_X_train, P_y_train, P_X_test, P_y_test = pickle.load(open('./cscw_data/P_train_and_test_label_text.pkl','rb'))
	X_train_feat, y_train, X_test_feat, y_test = pd.read_pickle(open('./cscw_data/P_train_and_test_label_feat.pkl', 'rb'))
	X_train_pos_feat = X_train_feat.iloc[np.where(P_y_train == 1)[0]]

	if args.alg == 'tsne':
		time_start = time.time()
		tsne = TSNE(n_jobs=4, n_components=2, verbose=1, perplexity=50, n_iter=1000, learning_rate=100)
		tsne_results = tsne.fit_transform(z)
		print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

		pickle.dump(tsne_results, open('./cscw_data/{}_z_tsne.pkl'.format(sample_data), 'wb'))

	elif args.alg == 'pca':
		time_start = time.time()
		pca = PCA(n_components=2)
		pca_results = pca.fit_transform(z)
		print(pca_results.shape, pca_results)
		print('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))

		pickle.dump(pca_results, open('./cscw_data/{}_z_pca.pkl'.format(sample_data), 'wb'))
