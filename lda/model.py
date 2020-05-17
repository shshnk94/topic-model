import argparse
from gensim.models import LdaModel
import sys
import pickle as pkl
import numpy as np
#import logging

from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from metrics import get_topic_coherence, get_topic_diversity, get_perplexity

parser = argparse.ArgumentParser(description='Latent Dirichlet Allocation')

parser.add_argument('--epochs', type=int, help='number of training epochs')
parser.add_argument('--batch_size', type=int, help='Batch size per training/test iteration')
parser.add_argument('--topics', type=int, help='number of topics to be trained on')
parser.add_argument('--data_path', type=str, help='path to the folder containing data')
parser.add_argument('--save_path', type=str, help='checkpoint path (if any)')

args = parser.parse_args()

def train(data, valid_h1, valid_h2, vocab):
    
    #logging.basicConfig(filename=args.save_path + 'lda.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = LdaModel(id2word=vocab,
                     num_topics=args.topics, 
                     random_state=0,
                     chunksize=args.batch_size,
                     alpha='auto',
                     per_word_topics=True)
    
    best_perplexity = float('inf') 

    for epoch in range(args.epochs):
        
        model.update(data, decay=0.5, offset=1, passes=1, update_every=0, eval_every=1, gamma_threshold=0.001, chunks_as_numpy=True)
        print("Epoch number " + str(epoch) + " and gensim ppl is " + str(model.log_perplexity(valid_h1)))

        val_perplexity = evaluate(data, valid_h1, valid_h2, model, 'valid')
        if val_perplexity < best_perplexity:
            best_perplexity = val_perplexity
            model.save(args.save_path + 'model_' + str(args.topics) + '_' + str(args.epochs) + '.ckpt')

def evaluate(train_set, test_h1, test_h2, model, step):

    theta = np.zeros((len(test_h1), args.topics))

    distribution = model.get_document_topics(test_h1, minimum_probability=0.0)
    for index, doc in enumerate(distribution):
        for topic, proportion in doc:
            theta[index, topic] = proportion 

    beta = model.get_topics()
    theta = np.array(theta)

    test = np.zeros((len(test_h2), beta.shape[1]))
    for index, doc in enumerate(test_h2):
        for word, count in doc:
            test[index, word] = count

    perplexity = get_perplexity(test, theta, beta)

    if step == 'test':

        distribution = model.show_topics(num_topics=args.topics, num_words=len(vocab), formatted=False)
        with open(args.save_path + 'topics.txt', 'w') as f:
            for topic in distribution:

                sorted_words = sorted(topic[1], key=lambda x: float(x[1]))
                f.write(' '.join([x for x, y in sorted_words[:10]]) + '\n')

        train_set = [np.expand_dims(np.array([word for word, count in doc if count != 0]), axis=0) for doc in train_set]
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

print("Evaluation on the test set")
model = LdaModel.load(args.save_path + 'model_' + str(args.topics) + '_' + str(args.epochs) + '.ckpt', mmap='r')
evaluate(train_set, test_h1, test_h2, model, 'test')
