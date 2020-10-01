from sklearn.decomposition import PCA, NMF
import numpy as np
import argparse
import os
import pickle as pkl

parser = argparse.ArgumentParser(description='PCA of beta matrix')

parser.add_argument('--topics', type=int, help='number of topics to be trained on')
parser.add_argument('--model', type=str, help='number of topics to be trained on')
parser.add_argument('--cvr', type=float, help='cumulative variance ratio that the components should express')
parser.add_argument('--data_path', type=str, help='path to the vocabulary')
parser.add_argument('--save_path', type=str, help='path to the beta matrix')

args = parser.parse_args()

with open(os.path.join(args.save_path, str(args.topics) + '_topics', 'beta.pkl'), 'rb') as f:
    beta = pkl.load(f)

with open(os.path.join(args.data_path, 'vocab.pkl'), 'rb') as f:
    vocab = pkl.load(f)

pca = PCA(n_components=args.cvr, svd_solver='full') if args.model == 'pca' else NMF(n_components=args.cvr)
pca.fit(beta.T)

print("{} determine 90% of the Cumulative Variance Ratio".format(pca.n_components_))

for component in range(pca.components_.shape[0]):

    topics = np.flip(np.argsort(pca.components_, axis=1), axis=1)[component]
    sorted_beta = np.flip(np.argsort(beta[topics[:10]], axis=1), axis=1)


"""
    for t in sorted_beta[:4]:
        print(','.join([vocab[w] for w in t[:10]]))
        
    print()
"""
