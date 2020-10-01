from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
import gensim
import numpy as np
import pickle as pkl
from scipy import sparse
import re
import string
import sys
import os
import argparse

from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

parser = argparse.ArgumentParser(description='Dimensionality reduction on pre-trained word embeddings.')

parser.add_argument('--topics', type=int, help='number of topics to be trained on')
parser.add_argument('--emb_path', type=str, help='path to the embedding file')
parser.add_argument('--save_path', type=str, default='./', help='checkpoint path (if any)')

args = parser.parse_args()

# Maximum / minimum document frequency
max_df = 0.7
min_df = 100  # choose desired value for min_df

# Read stopwords
with open('stops.txt', 'r') as f:
    stops = f.read().split('\n')

# Read data
print('reading data...')
train = fetch_20newsgroups(subset='train', shuffle=False)
test = fetch_20newsgroups(subset='test', shuffle=False)

train = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', train.data[doc]) for doc in range(len(train.data))]
test = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', test.data[doc]) for doc in range(len(test.data))]

def contains_punctuation(w):
    return any(char in string.punctuation for char in w)

def contains_numeric(w):
    return any(char.isdigit() for char in w)
    
# Remove documents with length less than 10 and greater than 95th percentile.
def remove_outlier(docs):
    lengths = np.array([len(doc) for doc in docs])
    docs = [docs[i] for i in np.where((lengths > 10) & (lengths < np.percentile(lengths, 95)))[0]]

    return docs

train = remove_outlier(train)
test = remove_outlier(test)
corpus = train + test

# Removes all words with any punctuation or digits in them.
corpus = [[w.lower() for w in corpus[doc] if not contains_punctuation(w)] for doc in range(len(corpus))]
corpus = [[w for w in corpus[doc] if not contains_numeric(w)] for doc in range(len(corpus))]
corpus = [[w for w in corpus[doc] if len(w)>1] for doc in range(len(corpus))]
corpus = [" ".join(corpus[doc]) for doc in range(len(corpus))]

# Create count vectorizer
print('counting document frequency of words...')
cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=None)
cvz = cvectorizer.fit_transform(corpus).sign()
print("Corpus size: ", len(corpus))
# Get vocabulary
print('building the vocabulary...')
sum_counts = np.asarray(cvz.sum(axis=0))[0]
v_size = sum_counts.shape[0]
print('initial vocabulary size: {}'.format(len(cvectorizer.vocabulary_)))

# Sort elements in vocabulary and also remove stop words from the list
vocab = sorted([(i, word) for i, word in enumerate(cvectorizer.vocabulary_) if word  not in stops], key=lambda x: x[0])
vocab = [w for i, w in vocab]
del cvectorizer

# Split in train/test/valid
print('tokenizing documents and splitting into train/test/valid...')
vaSize = int(0.2 * len(corpus))
trSize = len(corpus) - vaSize

#idx_permute = np.random.permutation(num_docs_tr).astype(int)
idx_permute = np.arange(len(corpus))

# Remove words not in train_data
vocab = list(set([w for idx_d in range(trSize) for w in corpus[idx_permute[idx_d]].split() if w in vocab]))
word2id = dict([(w, j) for j, w in enumerate(vocab)])
id2word = dict([(j, w) for j, w in enumerate(vocab)])
print('vocabulary after removing words not in train: {}'.format(len(vocab)))

# Write the vocabulary to a file
with open(os.path.join(args.save_path, 'vocab.pkl'), 'wb') as f:
    pkl.dump(vocab, f)

folder = '/'.join(args.emb_path.split('/')[:-1])
gensim.scripts.glove2word2vec.glove2word2vec(args.emb_path, os.path.join(folder, 'embeddings.txt'))
embeddings = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(folder, 'embeddings.txt'))

vectors = np.random.randn(len(vocab), 300)
for index, word in enumerate(vocab):
    if word in embeddings.wv.vocab:
        vectors[index] = embeddings[word]

svd =  TruncatedSVD(n_components = 50)
beta = svd.fit_transform(vectors)

with open('beta.pkl', 'wb') as f:
    pkl.dump(beta, f)
