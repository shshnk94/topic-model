import argparse
import ast
from collections import defaultdict

parser = argparse.ArgumentParser(description='string comparison based diversity calculation from file containing the topics')

parser.add_argument('--path', type=str, help='path to topics.txt')
parser.add_argument('--model', type=str, help='choose between etm/nvdm/prodlda')

args = parser.parse_args()

def parse():
    
    topics = []

    with open(args.path, 'r') as f:
        for line in f.readlines():

            if args.model == 'etm':
                topics.append(ast.literal_eval(line.split(':')[1].strip()))

            elif args.model == 'nvdm':
                pass
            elif args.model == 'prodlda':
                topics.append(line.strip().split())

    return topics


def diversity(topics):
    
    diversity = []
    related_topics = defaultdict(lambda: [])

    for i in range(len(topics)):
        for j in range(i+1, len(topics)):

            similarity = len(set(topics[i]).intersection(set(topics[j]))) / len(set(topics[i]).union(set(topics[j])))
            if similarity > 0.5:
                diversity.append(similarity)
                related_topics[i].append(j)

    return sum(diversity) / ((len(topics) - 1) * len(topics)), related_topics

topics = parse()
score, related_topics = diversity(topics)

print(1 - score, related_topics)
