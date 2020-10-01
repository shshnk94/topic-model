import argparse
import os
from gensim.models import LdaModel
import sys
import pickle as pkl
import numpy as np
#import logging

from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from metrics import get_topic_coherence, get_topic_diversity, get_perplexity

parser = argparse.ArgumentParser(description='Latent Dirichlet Allocation')

parser.add_argument('--mode', type=str, help='mode - train/test')
parser.add_argument('--fold', type=str, default='', help='current cross valid fold number')
parser.add_argument('--epochs', type=int, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size per training/test iteration')
parser.add_argument('--decay', type=float, default=0.5, help='weight what percentage of the previous lambda value is forgotten')
parser.add_argument('--offset', type=int, default=1, help='controls how much we will slow down the first steps the first few iterations')
parser.add_argument('--topics', type=int, help='number of topics to be trained on')
parser.add_argument('--data_path', type=str, help='path to the folder containing data')
parser.add_argument('--save_path', type=str, help='checkpoint path (if any)')

args = parser.parse_args()

if args.fold != '':

    args.data_path = os.path.join(args.data_path, 'fold{}'.format(args.fold))
    args.save_path = os.path.join(args.save_path, 'k{}_e{}_d{}_t{}'.format(args.topics, args.epochs, args.decay, args.offset), 'fold{}'.format(args.fold))

    if not os.path.isdir(args.save_path):
        os.system('mkdir -p ' + args.save_path)

ppl_file = open(os.path.join(args.save_path, 'val_scores.csv'), 'w')

def train(data, valid_h1, valid_h2, vocab):
    
    #logging.basicConfig(filename=args.save_path + 'lda.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = LdaModel(id2word=vocab,
                     num_topics=args.topics, 
                     random_state=0,
                     chunksize=args.batch_size,
                     update_every=args.batch_size,
                     alpha='auto',
                     eta=None,
                     decay=args.decay,
                     offset=args.offset,
                     per_word_topics=True)
    
    best_perplexity = float('inf') 

    for epoch in range(args.epochs):
        
        model.update(data, passes=1, eval_every=1, gamma_threshold=0.001)
        print("Epoch number {}".format(epoch), end=' ')

        val_perplexity = evaluate(data, valid_h1, valid_h2, model, 'valid')
        if val_perplexity < best_perplexity:
            best_perplexity = val_perplexity
            model.save(os.path.join(args.save_path, 'model.ckpt'))

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
    ppl_file.write(str(perplexity) + '\n')

    if step == 'test':

        distribution = model.show_topics(num_topics=args.topics, num_words=len(vocab), formatted=False)
        with open(args.save_path + 'topics.txt', 'w') as f:
            for topic in distribution:

                sorted_words = sorted(topic[1], key=lambda x: float(x[1]))
                f.write(' '.join([x for x, y in sorted_words[:10]]) + '\n')

        transformed_set = [np.expand_dims(np.array([word for word, count in doc if count != 0]), axis=0) for doc in train_set]
        coherence = get_topic_coherence(beta, transformed_set, 'lda')
        diversity = get_topic_diversity(beta)
        
        thetaWeightedAvg = np.zeros((1, args.topics))
        cnt = 0

        for base in range(0, len(train_set), args.batch_size):

            data_batch = train_set[base: min(base + args.batch_size, len(train_set))]
            sums = np.array([sum([count for word, count in doc]) for doc in data_batch])
            cnt += sums.sum(axis=0)

            theta = np.zeros((len(data_batch), args.topics))
            distribution = model.get_document_topics(data_batch, minimum_probability=0.0)
            for index, doc in enumerate(distribution):
                for topic, proportion in doc:
                    theta[index, topic] = proportion 

            weighed_theta = (theta.T * sums).T
            thetaWeightedAvg += np.expand_dims(weighed_theta.sum(axis=0), axis=0)

        thetaWeightedAvg = thetaWeightedAvg.squeeze() / cnt
        print('\nThe 10 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:10]))

        #Get top words in the topic
        for number, topic in model.show_topics(args.topics, 10, formatted=True):
            print([terms.split('*')[1].strip().strip('"') for terms in topic.split('+')])

    return perplexity

#Load training set and vocabulary
with open(os.path.join(args.data_path, 'vocab.new'), 'rb') as f:
    vocab = {index: word for index, word in enumerate(pkl.load(f))}

with open(os.path.join(args.data_path, 'train.pkl'), 'rb') as f:
    train_set = pkl.load(f)

if args.mode == 'train':

    with open(os.path.join(args.data_path, 'valid_h1.pkl'), 'rb') as f:
        valid_h1 = pkl.load(f)

    with open(os.path.join(args.data_path, 'valid_h2.pkl'), 'rb') as f:
        valid_h2 = pkl.load(f)

    model = train(train_set, valid_h1, valid_h2, vocab)

else:

    #Half held out testing
    with open(args.data_path + 'test_h1.pkl', 'rb') as f:
        test_h1 = pkl.load(f)

    with open(args.data_path + 'test_h2.pkl', 'rb') as f:
        test_h2 = pkl.load(f)

    print("Evaluation on the test set")
    model = LdaModel.load(os.path.join(args.save_path, 'model.ckpt'), mmap='r')
    evaluate(train_set, test_h1, test_h2, model, 'test')
