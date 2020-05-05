import argparse
from gensim.models import LdaModel
import sys
import pickle as pkl
import numpy as np

from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from metrics import get_topic_coherence, get_topic_diversity

parser = argparse.ArgumentParser(description='Latent Dirichlet Allocation')

#parser.add_argument('--mode', type=str, help='mode of the call - train/eval/test')
parser.add_argument('--epochs', type=int, help='number of training epochs')
parser.add_argument('--topics', type=int, help='number of topics to be trained on')
parser.add_argument('--dataset', type=str, default='20ng', help='name of corpus')
#parser.add_argument('--save_path', type=str, help='checkpoint path (if any)')

args = parser.parse_args()

#Load training, test data, and vocabulary
with open('lda/data/' + args.dataset + '/train.pkl', 'rb') as f:
    train = pkl.load(f)

with open('lda/data/' + args.dataset + '/test.pkl', 'rb') as f:
    valid = pkl.load(f)

with open('lda/data/' + args.dataset + '/vocab.new', 'rb') as f:
    vocab = pkl.load(f)

model = LdaModel(corpus=train,
                     num_topics=args.topics, 
                     random_state=0,
                     chunksize=64,
                     passes=args.epochs,
                     alpha='auto',
                     per_word_topics=True)

#Calculate metrics
print("Log perplexity: ", model.bound(valid))

probabilities = []
distribution = model.show_topics(num_topics=args.topics, num_words=len(vocab), formatted=False)
for topic in distribution:
    probabilities.append([x[1] for x in sorted(topic[1], key=lambda x: int(x[0]))])

probabilities = np.array(probabilities)
train = [np.expand_dims(np.array([word for word, count in doc if count != 0]), axis=0) for doc in train]

get_topic_coherence(probabilities, train, 'lda')
get_topic_diversity(probabilities, 'lda')
