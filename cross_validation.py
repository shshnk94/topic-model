import argparse
import os 
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(description = 'prodlda cross validation wrapper')

parser.add_argument('--model', type=str, help='choose between etm/nvdm/prodlda')
parser.add_argument('--dataset', type=str, default='20ng', help='name of corpus')
parser.add_argument('--topics', type=str, default='200', help='number of topics')
parser.add_argument('--data_path', type=str, help='path to a fold of data')
parser.add_argument('--save_path', type=str, help='save path for every run')
parser.add_argument('--gpu', type=str, help='index of the gpu core which would contain this model')

args = parser.parse_args()

def run_script(params, fold):
    
    if args.model == 'etm':

        train_embeddings = '1'
        os.system('CUDA_VISIBLE_DEVICES={} python etm/main.py --mode train'.format(args.gpu) + 
                  ' --dataset ' + args.dataset +
                  ' --fold ' + str(fold) + 
                  ' --batch_size 64' +
                  ' --data_path ' + args.data_path +
                  ' --save_path ' + args.save_path +
                  ' --num_topics ' + args.topics +
                  ' --train_embeddings ' + train_embeddings +
                  ' --epochs ' + params['epochs'] +
                  ' --lr ' + params['lr'] + ' &')

    elif args.model == 'prodlda':
        os.system('CUDA_VISIBLE_DEVICES={} python prodlda/run.py -m prodlda'.format(args.gpu) +
                  ' -f 100' + 
                  ' -s 100' +
                  ' -t ' + args.topics +
                  ' -b 64' + 
                  ' -r ' + params['lr'] +
                  ' -e ' + params['epochs'] +
                  ' --fold ' + str(fold) +
                  ' --mode train' +
                  ' --data_path ' + args.data_path + 
                  ' --save_path ' + args.save_path + ' &')

#Hyperparameters
hyperparameters = {'epochs': ['5000'],
                   'lr': ['2e-3']}#, '5e-4']}#, '5e-5']}

for params in ParameterGrid(hyperparameters):
    for fold in range(1): #Hard coded values of fold
        run_script(params, fold)
