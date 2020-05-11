import argparse
from gensim.models import LdaModel
import sys
import pickle as pkl
import numpy as np

from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from metrics import get_topic_coherence, get_topic_diversity, get_perplexity

parser = argparse.ArgumentParser(description='Latent Dirichlet Allocation')

#parser.add_argument('--mode', type=str, help='mode of the call - train/eval/test')
parser.add_argument('--epochs', type=int, help='number of training epochs')
parser.add_argument('--batch_size', type=int, help='Batch size per training/test iteration')
parser.add_argument('--topics', type=int, help='number of topics to be trained on')
parser.add_argument('--data_path', type=str, help='path to the folder containing data')
parser.add_argument('--save_path', type=str, help='checkpoint path (if any)')

args = parser.parse_args()

def data_loader(data, batch_size):

    indices = np.random.permutation(np.arange(len(data)))
    for base in range(0, indices.shape[0], batch_size):
        yield [data[i] for i in indices[base: min(base + batch_size, indices.shape[0])]]

def train(data, valid_h1, valid_h2, vocab):

    model = LdaModel(id2word=vocab,
                     num_topics=args.topics, 
                     random_state=0,
                     chunksize=args.batch_size,
                     alpha='auto',
                     per_word_topics=True)
    
    best_perplexity = float('inf') 

    for epoch in range(args.epochs):
        
        print("Epoch number", epoch, end=' ')
        for batch in data_loader(data, args.batch_size):
            model.update(batch, decay=0.5, offset=1, passes=1, update_every=0, gamma_threshold=0.001, chunks_as_numpy=True)

        val_perplexity = evaluate(data, valid_h1, valid_h2, model, 'valid')
        if val_perplexity < best_perplexity:
            best_perplexity = val_perplexity
            model.save(args.save_path + 'model.ckpt')

def evaluate(train_set, test_h1, test_h2, model, step):

    beta = []
    distribution = model.show_topics(num_topics=args.topics, num_words=len(vocab), formatted=False)

    with open(args.save_path + 'topics.txt', 'w') as f:
        for topic in distribution:

            sorted_words = sorted(topic[1], key=lambda x: float(x[1]))
            f.write(' '.join([x for x, y in sorted_words[:10]]) + '\n')
            beta.append([y for x, y in sorted_words])

    beta = np.array(beta)
    train_set = [np.expand_dims(np.array([word for word, count in doc if count != 0]), axis=0) for doc in train_set]
    
    theta = np.zeros((len(test_h1), args.topics))
    distribution = model.get_document_topics(test_h1, minimum_probability=0.0)
    for index, doc in enumerate(distribution):
        for topic, proportion in doc:
            theta[index, topic] = proportion 

    theta = np.array(theta)

    test = np.zeros((len(test_h2), beta.shape[1]))
    for index, doc in enumerate(test_h2):
        for word, count in doc:
            test[index, word] = count

    perplexity = get_perplexity(test, theta, beta)

    if step == 'test':
        coherence = get_topic_coherence(beta, train_set, 'lda')
        diversity = get_topic_diversity(beta, 'lda')

    return perplexity

#Load training set and vocabulary
with open(args.data_path + 'train.pkl', 'rb') as f:
    train_set = pkl.load(f)

with open(args.data_path + 'valid_h1.pkl', 'rb') as f:
    valid_h1 = pkl.load(f)

with open(args.data_path + 'valid_h2.pkl', 'rb') as f:
    valid_h2 = pkl.load(f)

with open(args.data_path + 'vocab.new', 'rb') as f:
    vocab = {index: word for index, word in enumerate(pkl.load(f))}

model = train(train_set, valid_h1, valid_h2, vocab)

#Half held out testing
with open(args.data_path + 'test_h1.pkl', 'rb') as f:
    test_h1 = pkl.load(f)

with open(args.data_path + 'test_h2.pkl', 'rb') as f:
    test_h2 = pkl.load(f)

model = LdaModel.load(args.save_path + 'model.ckpt', mmap='r')
evaluate(train_set, test_h1, test_h2, model, 'test')
