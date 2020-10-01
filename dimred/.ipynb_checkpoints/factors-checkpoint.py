from sklearn.decomposition import PCA
import argparse
import os
import pickle as pkl

parser = argparse.ArgumentParser(description='PCA of beta matrix')

parser.add_argument('--topics', type=int, help='number of topics to be trained on')
parser.add_argument('--data_path', type=str, help='path to the beta matrix')

args = parser.parse_args()

with open(os.path.join(args.data_path, 'beta.pkl'), 'rb') as f:
    beta = pkl.load(f)

pca = PCA()
pca.fit(beta)

print(pca.explained_variance_)
